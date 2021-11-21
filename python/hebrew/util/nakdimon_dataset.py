from typing import Tuple, List
import random
import numpy as np
import torch

from util import nakdimon_hebrew_model as hebrew
from util import nakdimon_utils as utils


class CharacterTable:
    MASK_TOKEN = ""

    def __init__(self, chars):
        # make sure to be consistent with JS
        self.chars = [CharacterTable.MASK_TOKEN] + chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def __len__(self):
        return len(self.chars)

    def to_ids(self, css):
        return [[self.char_indices[c] for c in cs] for cs in css]

    def to_str(self, ids):
        return [[self.indices_char[c] for c in cs] for cs in ids]

    def __repr__(self):
        return repr(self.chars)


letters_table = CharacterTable(hebrew.SPECIAL_TOKENS + hebrew.VALID_LETTERS)
dagesh_table = CharacterTable(hebrew.DAGESH)
sin_table = CharacterTable(hebrew.NIQQUD_SIN)
niqqud_table = CharacterTable(hebrew.NIQQUD)

LETTERS_SIZE = len(letters_table)
NIQQUD_SIZE = len(niqqud_table)
DAGESH_SIZE = len(dagesh_table)
SIN_SIZE = len(sin_table)


def print_tables():
    print("const ALL_TOKENS =", letters_table.chars, end=";\n")
    print("const niqqud_array =", niqqud_table.chars, end=";\n")
    print("const dagesh_array =", dagesh_table.chars, end=";\n")
    print("const sin_array =", sin_table.chars, end=";\n")


def from_categorical(t):
    return np.argmax(t, axis=-1)


def merge_unconditional(texts, tnss, nss, dss, sss):
    res = []
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            sentence.append(
                dagesh_table.indices_char[d] if hebrew.can_dagesh(t) else "\uFEFF"
            )
            sentence.append(
                sin_table.indices_char[s] if hebrew.can_sin(t) else "\uFEFF"
            )
            sentence.append(
                niqqud_table.indices_char[n] if hebrew.can_niqqud(t) else "\uFEFF"
            )
        res.append("".join(sentence))
    return res


class Data:
    def __init__(
        self,
        text=None,
        normalized=None,
        dagesh=None,
        sin=None,
        niqqud=None,
        device=None,
    ):
        self.text = text
        self.normalized = normalized
        self.dagesh = dagesh
        self.sin = sin
        self.niqqud = niqqud
        self.device = device
        # self.filenames = None

    @staticmethod
    def concatenate(others):

        device = others[0].normalized.device

        if "cpu" in str(device):
            text = np.concatenate([x.text for x in others])
            normalized = np.concatenate([x.normalized for x in others])
            dagesh = np.concatenate([x.dagesh for x in others])
            sin = np.concatenate([x.sin for x in others])
            niqqud = np.concatenate([x.niqqud for x in others])
        else:
            text = np.concatenate(
                [x.text for x in others]
            )  # torch.cat([x.text for x in others])
            normalized = torch.cat(
                [torch.tensor(x.normalized, device=device) for x in others]
            )
            dagesh = torch.cat([torch.tensor(x.dagesh, device=device) for x in others])
            sin = torch.cat([torch.tensor(x.sin, device=device) for x in others])
            niqqud = torch.cat([torch.tensor(x.niqqud, device=device) for x in others])

        return Data(text, normalized, dagesh, sin, niqqud)

    def shapes(self):
        return (
            self.text.shape,
            self.normalized.shape,
            self.dagesh.shape,
            self.sin.shape,
            self.niqqud.shape,
        )  # , self.kind.shape

    def size(self):
        self.shapes()[0][0]

    def shuffle(self):
        utils.shuffle_in_unison(
            self.text, self.normalized, self.dagesh, self.niqqud, self.sin
        )

    def to_device(self, device):
        self.normalized = torch.tensor(self.normalized).to(device)
        self.niqqud = torch.tensor(self.niqqud).to(device)
        self.dagesh = torch.tensor(self.dagesh).to(device)
        self.sin = torch.tensor(self.sin).to(device)
        self.device = device

    def get_idces(self, idces):

        if type(idces) == int:
            idces = [idces]

        return Data(
            self.text[idces],
            self.normalized[idces],
            self.dagesh[idces],
            self.sin[idces],
            self.niqqud[idces],
            self.device,
        )

    def __getitem__(self, items):
        return self.get_idces(items)

    @staticmethod
    def from_text(heb_items, maxlen: int) -> "Data":
        assert heb_items
        text, normalized, dagesh, sin, niqqud = zip(
            *(zip(*line) for line in hebrew.split_by_length(heb_items, maxlen))
        )

        def pad(ords, dtype="int32", value=0):
            return utils.pad_sequences(ords, maxlen=maxlen, dtype=dtype, value=value)

        normalized = pad(letters_table.to_ids(normalized))
        dagesh = pad(dagesh_table.to_ids(dagesh))
        sin = pad(sin_table.to_ids(sin))
        niqqud = pad(niqqud_table.to_ids(niqqud))
        text = pad(text, dtype="<U1", value=0)

        return Data(text, normalized, dagesh, sin, niqqud)

    def __len__(self):
        return self.normalized.shape[0]

    def print_stats(self):
        print(self.shapes())


def read_corpora(base_paths):
    return tuple(
        [
            (filename, list(hebrew.iterate_file(filename)))
            for filename in list(utils.iterate_files(base_paths))
        ]
    )


def load_data(
    corpora, validation_rate: float, maxlen: int, shuffle=True, subtraining_rate=1
) -> Tuple[Data, Data]:

    corpus = [
        (filename, Data.from_text(heb_items, maxlen))
        for (filename, heb_items) in corpora
    ]

    validation_data = None
    if validation_rate > 0:
        np.random.shuffle(corpus)
        size = sum(len(x) for _, x in corpus)
        validation_size = size * validation_rate
        validation = []
        validation_filenames: List[str] = []
        total_size = 0
        while total_size < validation_size:
            if abs(total_size - validation_size) < abs(
                total_size + len(corpus[-1]) - validation_size
            ):
                break
            (filename, c) = corpus.pop()
            total_size += len(c)
            validation.append(c)
            validation_filenames.append(filename)
        validation_data = Data.concatenate(validation)
        validation_data.filenames = tuple(validation_filenames)

    cs = [c for (_, c) in corpus]
    random.shuffle(cs)
    train = Data.concatenate(cs[: int(subtraining_rate * len(corpus))])
    if shuffle:
        train.shuffle()
    return train, validation_data


def get_data(l_filenames, max_len, validation_rate=0.0):

    train, test = load_data(
        tuple(read_corpora(tuple(l_filenames))),
        validation_rate=validation_rate,
        maxlen=max_len,
    )

    return train, test
