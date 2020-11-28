from entityMaker.helper import make_days, make_hour, make_minute, make_second, make_position
from entityMaker.helper import make_timestamp_date, make_timestamp_clock, make_timestamp_word


def generate_RouterWiFiDuration(language, amount_minute=None, amount_hour=None):
    entity_list = []
    entity_type = "RouterWiFiDuration"
    entity_list += make_minute(entity_type, language, amount=amount_minute)
    entity_list += make_hour(entity_type, language, amount=amount_hour)
    return entity_list


def generate_Duration(language, amount_second=None, amount_minute=None, amount_hour=None):
    entity_list = []
    entity_type = "Duration"
    entity_list += make_second(entity_type, language, amount=amount_second)
    entity_list += make_minute(entity_type, language, amount=amount_minute)
    entity_list += make_hour(entity_type, language, amount=amount_hour)
    return entity_list


def generate_RadioChannelPosition(language, amount=None, max_range=100):
    entity_list = []
    entity_type = "RadioChannelPosition"
    entity_list += make_position(entity_type, language, amount=amount, max_range=max_range)
    return entity_list


def generate_TvChannelPosition(language, amount=None, max_range=100):
    entity_list = []
    entity_type = "TvChannelPosition"
    entity_list += make_position(entity_type, language, amount=amount, max_range=max_range)
    return entity_list


def generate_LocalsearchTimeStampStartTime(language, amount=None):
    entity_list = []
    entity_type = "LocalsearchTimeStampStartTime"
    entity_list += make_timestamp_clock(entity_type, language, amount=amount)
    return entity_list


def generate_LocalsearchTimeStampEndTime(language, amount=None):
    entity_list = []
    entity_type = "LocalsearchTimeStampEndTime"
    entity_list += make_timestamp_clock(entity_type, language, amount=amount)
    return entity_list


def generate_LocalsearchTimeStampStartDay(language, amount_date=None, amount_word=None):
    entity_list = []
    entity_type = "LocalsearchTimeStampStartDay"
    entity_list += make_timestamp_date(entity_type, language, amount=amount_date)
    entity_list += make_timestamp_word(entity_type, language, amount=amount_word)
    return entity_list
