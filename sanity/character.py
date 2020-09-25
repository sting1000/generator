"""
(c) Copyright 2020 Swisscom AG
All Rights Reserved.
"""

# Taken from https://git.swisscom.com/projects/EELV/repos/transformer-model/browse/sanity?at=refs%2Fheads%2Fdevelop

from sanity.config import valid_chars


def check(cleaned_transcription):
    for char in cleaned_transcription:
        if char not in valid_chars:
            return 'ILLEGAL_CHARACTER({})'.format(char)
    return True
