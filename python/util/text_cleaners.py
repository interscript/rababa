import re
from util.constants import VALID_ARABIC, BASIC_HARAQAT, ALL_POSSIBLE_HARAQAT
from diacritization_evaluation import util

_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    text = re.sub(_whitespace_re, " ", text)
    return text

def basic_cleaners(text):
    text = collapse_whitespace(text)
    return text.strip()

def valid_arabic_cleaners(text):
    text = filter(lambda char: char in VALID_ARABIC, text)
    text = collapse_whitespace(''.join(list(text)))
    return text.strip()

def extract_stack(stack, correct_reversed: bool = True):
    """
    Given stack, we extract its content to string, and check whether this string is
    available at all_possible_haraqat list: if not we raise an error. When correct_reversed
    is set, we also check the reversed order of the string, if it was not already correct.
    """
    char_haraqat = []
    while len(stack) != 0:
        char_haraqat.append(stack.pop())
    full_haraqah = "".join(char_haraqat)
    reversed_full_haraqah = "".join(reversed(char_haraqat))
    if full_haraqah in ALL_POSSIBLE_HARAQAT:
        out = full_haraqah
    elif reversed_full_haraqah in ALL_POSSIBLE_HARAQAT and correct_reversed:
        out = reversed_full_haraqah
    else:
        #raise ValueError(stack)

        #raise ValueError(
        #    f"""The chart has the following haraqat which are not found in
        #all possible haraqat: {'|'.join([ALL_POSSIBLE_HARAQAT[diacritic]
        #                                 for diacritic in full_haraqah ])}"""
        #)
        out = ''
    return out

def extract_haraqat(text: str, correct_reversed: bool = True):
    """
    Args:
    text (str): text to be diacritized
    Returns:
    text: the original text as it comes
    text_list: all text that are not haraqat
    haraqat_list: all haraqat_list
    """
    if len(text.strip()) == 0:
        return text, [" "] * len(text), [""] * len(text)
    stack = []
    haraqat_list = []
    txt_list = []
    for char in text:
        # if chart is a diacritic, then extract the stack and empty it
        if char not in BASIC_HARAQAT.keys():
            stack_content = extract_stack(stack,
                                          correct_reversed=correct_reversed)
            #if stack_content != '':
            haraqat_list.append(stack_content)
            txt_list.append(char)
            stack = []
        else:
            stack.append(char)
    if len(haraqat_list) > 0:
        del haraqat_list[0]
    haraqat_list.append(extract_stack(stack))

    return text, txt_list, haraqat_list
