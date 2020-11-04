import unicodedata
import re


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([a-zA-Z]+)[\:\-\'\.]", r"\1 ", s)  # colun: dsfa -> colun dsfa
    s = re.sub(r"[!\"#&'()*,\-;<=>?@_`~]", r" ", s)  # colun-dsfa -> colun dsfa
    s = re.sub(r"([\+])", r" \1 ", s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s


def remove_noisy_tags(text: str) -> str:
    """
    Removes the CH, D, F, I, HD tags from the string.
    """
    text = re.sub(r'\b(?:)(CH|DE|FR|F|HD|UHD)\b', '', text, flags=re.IGNORECASE)
    return text
