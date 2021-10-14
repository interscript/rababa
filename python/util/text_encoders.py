from util import text_cleaners
from typing import Dict, List, Optional
from util.constants import ALL_POSSIBLE_HARAQAT, decision_fct


class TextEncoder:
    pad = "P"

    def __init__(
        self,
        input_chars: List[str],
        target_model: Dict[str, str],
        cleaner_fn: Optional[str] = None,
        reverse_input: bool = False,
        reverse_target: bool = False,
    ):
        if cleaner_fn:
            self.cleaner_fn = getattr(text_cleaners, cleaner_fn)
        else:
            self.cleaner_fn = None

        self.target_model = target_model

        self.input_symbols: List[str] = [TextEncoder.pad] + input_chars

        # self.target_symbols: List[str] = [TextEncoder.pad] + target_chars

        self.d_target_symbols = dict([(k, [TextEncoder.pad]+self.target_model[k])
                                       for k in target_model.keys()])

        self.input_symbol_to_id: Dict[str, int] = {
            s: i for i, s in enumerate(self.input_symbols)
        }
        self.d_input_id_to_symbol = [(k, {s: i for i, s in enumerate(self.input_symbols)})
                                     for k in self.target_model.keys()]
        self.input_id_to_symbol: Dict[int, str] = {
            i: s for i, s in enumerate(self.input_symbols)
        }

        # self.target_symbol_to_id: Dict[str, int] = {
        #     s: i for i, s in enumerate(self.target_symbols)
        # }
        # self.target_id_to_symbol: Dict[int, str] = {
        #    i: s for i, s in enumerate(self.target_symbols)
        # }

        self.d_target_symbol_to_id = dict([(k, {s: i for i, s in enumerate(self.d_target_symbols[k])})
                                           for k in self.d_target_symbols.keys()])

        self.d_target_id_to_symbol = dict([(k, {i: s for i, s in enumerate(self.d_target_symbols[k])})
                                           for k in self.d_target_symbols.keys()])

        self.reverse_input = reverse_input
        self.reverse_target = reverse_target
        self.input_pad_id = self.input_symbol_to_id[self.pad]

        # self.target_pad_id = self.target_symbol_to_id[self.pad]
        self.d_target_pad_id = dict([(k, self.d_target_symbol_to_id[k][self.pad])
                                     for k in self.d_target_symbols.keys()])
        self.start_symbol_id = None


    def input_to_sequence(self, text: str) -> List[int]:
        if self.reverse_input:
            text = "".join(list(reversed(text)))

        sequence = [self.input_symbol_to_id[s] for s in text
                    if s not in [self.pad] and \
                    self.input_symbol_to_id.get(s, False)]
        if len(sequence) == 0:
            # handle cases with zero length strings (no arabic symbols)
            sequence = [self.input_symbol_to_id[s] for s in ' ']

        return sequence

    def target_to_sequence(self, text: str) -> List[int]:
        if self.reverse_target:
            text = "".join(list(reversed(text)))
        #sequence = [self.target_symbol_to_id[s] for s in text if s not in [self.pad]]
        d_sequence = {} # []
        for k in self.input_symbols.keys():
            d_sequence[k] = [self.d_target_symbol_to_id[k][s]
                             for s in text if s not in [self.pad]]
        return d_sequence

    def sequence_to_input(self, sequence: List[int]):
        return [
            self.input_id_to_symbol[symbol]
            for symbol in sequence
            if symbol in self.input_id_to_symbol and symbol not in [self.input_pad_id]
        ]

    def sequence_to_target(self, d_sequence): #List[int]):
        """
        return [
            self.target_id_to_symbol[symbol]
            for symbol in sequence
            if symbol in self.target_id_to_symbol and symbol not in [self.target_pad_id]
        ]
        """
        d_target = {}
        for k in self.input_symbols.keys():
            d_target[k] = [self.d_target_id_to_symbol[symbol]
                           for symbol in sequence
                           if symbol in self.d_target_id_to_symbol[k] and symbol not in [self.target_pad_id]]
        return d_target

    def clean(self, text):
        if self.cleaner_fn:
            return self.cleaner_fn(text)
        return text

    def combine_text_and_haraqat(self, input_ids, d_output_ids):
        # combine_text_and_haraqat(self, input_ids: List[int], output_ids: List[int]):
        """
        Combines the  input text with its corresponding  haraqat
        Args:
            inputs: a list of ids representing the input text
            outputs: a list of ids representing the output text
        Returns:
        text: the text after merging the inputs text representation with the output
        representation
        """
        output = ""
        for i, input_id in enumerate(input_ids):
            if input_id == self.input_pad_id:
                break

            output += self.d_input_id_to_symbol[input_id]
            # if d_output_ids['shaddah'] !=
            d_dia = dict((k, self.d_target_id_to_symbol[k][d_output_ids[k][i]])
                          for k in self.input_symbols.keys())
            output += decision_fct(d_dia)
        return output

    def __str__(self):
        return type(self).__name__


class BasicArabicEncoder(TextEncoder):
    def __init__(
        self,
        cleaner_fn="basic_cleaners",
        reverse_input: bool = False,
        reverse_target: bool = False,
    ):
        input_chars: List[str] = list("بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث")
        #target_chars: List[str] = list(ALL_POSSIBLE_HARAQAT.keys())
        target_model = {
                'fatha': ["", "َ"],
                'shaddah': ["", "ّ"],
                'haraqat': ["", "ً", "ُ", "ٌ", "ِ", "ٍ", "ْ"]
                }

        super().__init__(
            input_chars,
            target_model,
            cleaner_fn=cleaner_fn,
            reverse_input=reverse_input,
            reverse_target=reverse_target,
        )


class ArabicEncoderWithStartSymbol(TextEncoder):
    def __init__(
        self,
        cleaner_fn="basic_cleaners",
        reverse_input: bool = False,
        reverse_target: bool = False,
    ):
        input_chars: List[str] = list("بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث")
        # the only difference from the basic encoder is adding the start symbol
        #target_chars: List[str] = list(ALL_POSSIBLE_HARAQAT.keys()) + ["s"]
        target_model = {
                'haraqat': ["", "ً", "ُ", "ٌ", "ِ", "ٍ", "ْ"],
                'shaddah': ["", "ّ"],
                'fatha': ["", "َ"]
                }

        super().__init__(
            input_chars,
            target_model,
            cleaner_fn=cleaner_fn,
            reverse_input=reverse_input,
            reverse_target=reverse_target,
        )

        #self.start_symbol_id = self.d_target_symbol_to_id['haraqat']["s"]
