import unicodedata
import re


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def clean_string(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([a-zA-Z]+)[\:\-\'\.]", r"\1 ", s)  # colun: dsfa -> colun dsfa
    s = re.sub(r'(\d+)[\:.] ', r'\1 ', s)
    s = re.sub(r"[!\"#'()*,\-;<=>?@_`~|]", r" ", s)  # colun-dsfa -> colun dsfa
    s = re.sub(r"([\+])", r" \1 ", s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r' [.\-:] ', ' ', s)
    s = re.sub(r'(\d+)[\:.] ', r'\1 ', s)  # a. -> a
    s = re.sub(r'\.{2,}', r' ', s)  # ... -> ''
    s = re.sub(r'\: ', r' ', s) # 2: -> ' '
    #TODO: 123.abc -> 123 abc
    s = s.strip()
    return s


def remove_noisy_tags(text: str) -> str:
    """
    Removes the CH, D, F, I, HD tags from the string.
    """
    text = re.sub(r'\b(?:)(CH|DE|FR|EN|F|HD|UHD)\b', '', text, flags=re.IGNORECASE)
    return text
