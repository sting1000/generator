import re
from src.command_generator.Normalizer import Normalizer




def remove_sharp_sign(sentence: str) -> str:
    regex = r'#[a-zA-Z]*'
    search = re.search(regex, sentence)
    while search:
        pos = search.start()
        sentence = sentence[:pos] + sentence[(pos + 1):]
        search = re.search(regex, sentence)
    return sentence


def assign_tag_to_words(sentence: str, tag: str) -> str:
    regex = r"(?!{\w*)\w+\.?\:?\w*(?!\w*})"
    result = re.sub(regex, tag, sentence, 0, re.MULTILINE)
    return result


def make_bio_tag(value_norm, tag) -> list:
    """
    assign bio tags for number related entities
    """
    bio_tags = []
    vsb_list = ["VodName", "SeriesName", "BroadcastName"]
    num_list = ["Duration", "RouterWiFiDuration", "TvChannelPosition", "RadioChannelPosition",
                "LocalsearchTimeStampStartDay"]
    time_list = ["LocalsearchTimeStampEndTime", "LocalsearchTimeStampStartTime"]

    if tag in vsb_list:
        appendix = 'name'
    elif tag in num_list:
        appendix = 'number'
    elif tag in time_list:
        appendix = 'time'
    else:
        appendix = 'O'
        return [appendix] * len(value_norm)

    bio_tags.append('B_' + appendix)
    bio_tags.extend(['I_' + appendix] * (len(value_norm) - 1))
    return bio_tags
