import numpy
import re

def filter_aliases(aliases: list) -> list:
    """
    Filters list of aliases to keep only the useful ones.
    It is used to remove all the noisy aliases given by tv that are useful for ASR.
    
    E.g. ['s r f 1', 'SRF 1'] becomes ['SRF 1']
    """
    original_aliases = list(aliases)
    regex = re.compile(r'\b[a-zA-Z]\b')
    for alias in original_aliases:
        if regex.findall(alias):  #modified 
            uppercased_alias = restore_abbreviations_in_text(text=alias, uppercase=True).strip()
            aliases.remove(alias)
            if uppercased_alias not in aliases:
                aliases.append(uppercased_alias)
    return list(set(aliases))

def restore_abbreviations_in_text(text: str, uppercase=False) -> str:
    """
    Restores malformed abbreviations in text.
    E.g. 'Go to S R F 1' becomes 'Go to SRF 1'.
    """
    abbreviations = find_space_separated_abbreviations(text=text)
    if abbreviations:
        for abbreviation in abbreviations:
            if uppercase:
                text = text.replace(' '.join(list(abbreviation)), abbreviation.upper())
            else:
                text = text.replace(' '.join(list(abbreviation)), abbreviation)
    return text

def find_space_separated_abbreviations(text: str) -> list:
    """
    Finds abbreviations in text written with space among their letters.
    E.g. 'Go to S R F 1' finds 'SRF' as abbreviation.
    """
    regex = re.compile(r'\b[a-zA-Z]\b')

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

