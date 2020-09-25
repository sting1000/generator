"""
(c) Copyright 2020 Swisscom AG
All Rights Reserved.
"""

# Taken from https://git.swisscom.com/projects/EELV/repos/transformer-model/browse/sanity?at=refs%2Fheads%2Fdevelop

from sanity.character import check as check_character
from sanity.foreignword import check as check_foreignword, clean as clean_foreignwords
from sanity.tag import check as check_tag, clean as clean_tags
from sanity.whitespace import check as check_whitespace


def check(transcription):
    tag_check_result = check_tag(transcription)
    if tag_check_result is not True:
        return tag_check_result

    foreignword_check_result = check_foreignword(transcription)
    if foreignword_check_result is not True:
        return foreignword_check_result

    whitespace_check_result = check_whitespace(transcription)
    if whitespace_check_result is not True:
        return whitespace_check_result

    # Next we have to clean the transcription (remove tags, foreign words, etc.)
    # This is needed to do character validation
    cleaned = clean_tags(transcription)
    cleaned = clean_foreignwords(cleaned)

    character_check_result = check_character(cleaned)
    if character_check_result is not True:
        return character_check_result

    return True
