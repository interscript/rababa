
from typing import Tuple, List
from pathlib import Path

import numpy as np

import hebrew


basepath = Path('tests/validation/expected')


def metric_cha(actual: str, expected: str, *args, **kwargs) -> float:
    """
    Calculate character-level agreement between actual and expected.
    """
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)
    return mean_equal((x, y) for x, y in zip(actual_hebrew, expected_hebrew)
                      if hebrew.can_any(x.letter))


def metric_dec(actual: str, expected: str, *args, **kwargs) -> float:
    """
    Calculate nontrivial-decision agreement between actual and expected.
    """
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)

    return mean_equal(
       ((x.niqqud, y.niqqud) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_niqqud(x.letter)),

       ((x.dagesh, y.dagesh) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_dagesh(x.letter)),

       ((x.sin, y.sin) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_sin(x.letter)),
    )


def is_hebrew(token):
    return len([c for c in token.items if c.letter in hebrew.HEBREW_LETTERS]) > 1


def metric_wor(actual: str, expected: str, *args, **kwargs) -> float:
    """
    Calculate token-level agreement between actual and expected, for tokens containing at least 2 Hebrew letters.
    """
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)
    actual_tokens = hebrew.tokenize(actual_hebrew)
    expected_tokens = hebrew.tokenize(expected_hebrew)
    # for x, y in zip(actual_tokens, expected_tokens):
    #     if is_hebrew(x) and x != y and '"' not in token_to_text(x):
    #         print('מצוי', token_to_text(x))
    #         print('רצוי', token_to_text(y))
    #         print()
    return mean_equal((x, y) for x, y in zip(actual_tokens, expected_tokens)
                      if is_hebrew(x))


def mean_equal(*pair_iterables):
    total = 0
    acc = 0
    for pair_iterable in pair_iterables:
        pair_iterable = list(pair_iterable)
        total += len(pair_iterable)
        acc += sum(x == y for x, y in pair_iterable)
    return acc / total


def get_diff(actual, expected):
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            return f'\n{actual[i-15:i+15]}\n!=\n{expected[i-15:i+15]}'
    return ''


def get_items(actual: str, expected: str, vocalize=False) -> Tuple[List[hebrew.HebrewChar], List[hebrew.HebrewChar]]:
    expected_hebrew = list(hebrew.iterate_dotted_text(expected))
    actual_hebrew = list(hebrew.iterate_dotted_text(actual))
    if vocalize:
        expected_hebrew = [x.vocalize() for x in expected_hebrew]
        actual_hebrew = [x.vocalize() for x in actual_hebrew]
    diff = get_diff(repr(''.join(c.letter for c in actual_hebrew)),
                    repr(''.join(c.letter for c in expected_hebrew)))
    assert not diff, diff
    return actual_hebrew, expected_hebrew


def split_to_sentences(text):
    return [sent + '.' for sent in text.split('. ') if len(hebrew.remove_niqqud(sent)) > 15]


def clean_read(filename):
    with open(filename, encoding='utf8') as f:
        return cleanup(f.read())


def all_diffs_for_files(expected_filename, system1, system2):
    expected_sentences = split_to_sentences(clean_read(expected_filename))
    actual_sentences1 = split_to_sentences(clean_read(expected_filename.replace('expected', system1)))
    actual_sentences2 = split_to_sentences(clean_read(expected_filename.replace('expected', system2)))
    assert len(expected_sentences) == len(actual_sentences1) == len(actual_sentences2)

    triples = [(e, a1, a2) for (e, a1, a2) in zip(expected_sentences, actual_sentences1, actual_sentences2)
               if metric_wor(a1, e) < 0.90 or metric_wor(a2, e) < 0.90]
    triples.sort(key=lambda e_a1_a2: metric_cha(e_a1_a2[2], e_a1_a2[0]))
    for (e, a1, a2) in triples[:20]:
        print(f"{system1}: {metric_wor(a1, e):.2%}; {system2}: {metric_wor(a2, e):.2%}")
        print('סבבה:', a1)
        print('מקור:', e)
        print('גרוע:', a2)
        print()


def all_diffs(system1, system2):
    for folder in basepath.iterdir():
        for file in folder.iterdir():
            all_diffs_for_files(str(file), system1, system2)


def collect_failed_words_for_files(system):
    for folder in basepath.iterdir():
        for file in folder.iterdir():
            expected_filename = str(file)
            expected_sentences = split_to_sentences(clean_read(expected_filename))
            actual_sentences = split_to_sentences(clean_read(expected_filename.replace('expected', system)))
            assert len(expected_sentences) == len(actual_sentences)

            actual_tokens = [token for sentence in actual_sentences for token in sentence.split()]
            expected_tokens = [token for sentence in expected_sentences for token in sentence.split()]
            assert len(actual_tokens) == len(expected_tokens)
            yield from [(x, y) for x, y in zip(expected_tokens, actual_tokens) if x != y]


def all_metrics(actual, expected):
    return {'dec': metric_dec(actual, expected),
            'cha': metric_cha(actual, expected),
            'wor': metric_wor(actual, expected),
            'voc': metric_wor(actual, expected, vocalize=True)}


def cleanup(text):
    return ' '.join(text.strip().split())


def all_metrics_for_files(actual_filename, expected_filename):
    with open(expected_filename, encoding='utf8') as f:
        expected = cleanup(f.read())

    with open(actual_filename, encoding='utf8') as f:
        actual = cleanup(f.read())
    try:
        return all_metrics(actual, expected)
    except AssertionError as ex:
        raise RuntimeError(actual_filename) from ex
