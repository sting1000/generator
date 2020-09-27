import numpy
import re

def filter_aliases(aliases: list) -> list:
    """
    Filters list of aliases to keep only the useful ones.
    It is used to remove all the noisy aliases given by tv that are useful for ASR.
    
    E.g. ['s r f 1', 'SRF 1'] becomes ['SRF 1']
    """
    original_aliases = list(aliases)
    regex = re.compile(r'\b[a-z]\b')
    set_lowercased_aliases = set([alias.lower() for alias in aliases])
    for alias in original_aliases:
        if alias.islower() and regex.finditer(alias):
            uppercased_alias = alias.upper()
            uppercased_alias = restore_abbreviations_in_text(text=uppercased_alias).strip()
            if uppercased_alias.lower() in set_lowercased_aliases:
                aliases.remove(alias)
    return aliases

def find_space_separated_abbreviations(text: str) -> list:
    """
    Finds abbreviations in text written with space among their letters.
    E.g. 'Go to S R F 1' finds 'SRF' as abbreviation.
    """
    regex = re.compile(r'\b[A-Z]\b')

    # initialize values
    abbreviations = []
    abbreviation = ''
    last_pos = -1
    
    for item in regex.finditer(text):
        if last_pos == -1:
            abbreviation += item.group()
            last_pos = item.span()[1]
        elif item.span()[0] == last_pos + 1:
            abbreviation += item.group()
            last_pos = item.span()[1]
        elif len(abbreviation) > 1:
            abbreviations.append(abbreviation)
            abbreviation = item.group()
            last_pos = -1
        elif len(abbreviation) == 1:
            abbreviation = item.group()
            last_pos = -1
    
    # append last found abbreviation
    if len(abbreviation) > 1:
        abbreviations.append(abbreviation)

    return abbreviations

def restore_abbreviations_in_text(text: str) -> str:
    """
    Restores malformed abbreviations in text.
    E.g. 'Go to S R F 1' becomes 'Go to SRF 1'.
    """
    abbreviations = find_space_separated_abbreviations(text=text)
    if abbreviations:
        for abbreviation in abbreviations:
            text = text.replace(' '.join(list(abbreviation)), abbreviation)
    return text

