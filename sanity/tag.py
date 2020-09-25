"""
(c) Copyright 2020 Swisscom AG
All Rights Reserved.
"""

# Taken from https://git.swisscom.com/projects/EELV/repos/transformer-model/browse/sanity?at=refs%2Fheads%2Fdevelop

import re

from sanity.config import valid_tag_names, valid_seperate_tag_names

TAG_REGEX = re.compile('\\[.*?\\]')
CONCAT_TAG_REGEX = re.compile('\\[.*?\\]\\S')


def check(transcription):
    tags = get_all_tags(transcription)

    check_tag_whitespaces_result = check_tag_whitespaces(tags)
    if check_tag_whitespaces_result is not True:
        return check_tag_whitespaces_result

    check_tag_names_result = check_tag_names(tags)
    if check_tag_names_result is not True:
        return check_tag_names_result

    check_seperate_tags_result = check_seperate_tags(tags, transcription)
    if check_seperate_tags_result is not True:
        return check_seperate_tags_result

    check_concatenated_tags_result = check_concatenated_tags(transcription)
    if check_concatenated_tags_result is not True:
        return check_concatenated_tags_result

    return True


def clean(transcription):
    tags = get_all_tags(transcription)
    cleaned_transcription = transcription

    for tag in tags:
        cleaned_transcription = cleaned_transcription.replace('[{}]'.format(tag), '')

    return cleaned_transcription


def get_all_tags(transcription):
    tags = []
    for match in re.finditer(TAG_REGEX, transcription):
        whole_match = match.group(0)
        tagName = whole_match[1:-1]
        tags.append(tagName)
    return tags


def check_tag_names(tags):
    for tag in tags:
        if tag not in valid_tag_names:
            return 'CUSTOM_TAG_NAME({})'.format(tag)

    return True


def check_seperate_tags(tags, transcription):
    for tag in tags:
        if tag in valid_seperate_tag_names:
            whole_tag = '[{}]'.format(tag)
            if not whole_tag == transcription:
                return 'SEPERATE_TAG_IN_TEXT({})'.format(tag)

    return True


def check_concatenated_tags(transcription):
    m = re.findall(CONCAT_TAG_REGEX, transcription)
    if m:
        return 'NO_WHITE_SPACE_AFTER_TAG_IN_TEXT({})'.format(m)
    return True


def check_tag_whitespaces(tags):
    for tag in tags:
        if ' ' in tag:
            return 'WHITESPACE_IN_TAG({})'.format(tag)

    return True
