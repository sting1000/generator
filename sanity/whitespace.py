"""
(c) Copyright 2020 Swisscom AG
All Rights Reserved.
"""


# Taken from https://git.swisscom.com/projects/EELV/repos/transformer-model/browse/sanity?at=refs%2Fheads%2Fdevelop

def check(transcription):
    if transcription.startswith(' '):
        return 'ILLEGAL_WHITESPACE(START)'
    if transcription.endswith(' '):
        return 'ILLEGAL_WHITESPACE(END)'
    if '  ' in transcription:
        return 'ILLEGAL_WHITESPACE(DOUBLE)'
    return True
