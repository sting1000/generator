"""
(c) Copyright 2020 Swisscom AG
All Rights Reserved.
"""

# Taken from https://git.swisscom.com/projects/EELV/repos/transformer-model/browse/sanity?at=refs%2Fheads%2Fdevelop

import re

FOREIGN_WORD_REGEX = re.compile('#.*?($| )')


def check(transcription):
    cleaned = clean(transcription)

    if '#' in cleaned:
        return 'INVALID_FOREIGNWORD({})'.format(cleaned)

    return True


def clean(transcription):
    words = get_all_foreignwords(transcription)

    # Remove all valid occurrences from the transcription
    cleaned_transcription = transcription

    for word in words:
        cleaned_transcription = cleaned_transcription.replace('#' + word, word)  # just remove the hashtag

    return cleaned_transcription


def get_all_foreignwords(transcription):
    words = []
    for match in re.finditer(FOREIGN_WORD_REGEX, transcription):
        whole_match = match.group(0)

        if whole_match.endswith(' '):
            word = whole_match[1:-1]
        else:
            word = whole_match[1:]

        if word == '':
            continue

        words.append(word)
    return words
