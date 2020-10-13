from entityMaker.helper import make_days, make_hours, make_minutes, make_seconds, make_positions
from entityMaker.helper import make_timestamp_date, make_timestamp_clock, make_timestamp_word
import random


def maker_RouterWiFiDuration(language_list=['de', 'en'], amount=None, entity_amount=None, random_seed=42):
    entity_list = []
    random.seed(random_seed)
    entity_type = "RouterWiFiDuration"
    for lan in language_list:
        entity_list += make_minutes(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
        entity_list += make_hours(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
        entity_list += make_days(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
    if amount is None:
        amount = len(entity_list)
    return random.choices(entity_list, k=amount)


def maker_Duration(language_list=['de', 'en'], amount=None, entity_amount=None, random_seed=42):
    entity_list = []
    random.seed(random_seed)
    entity_type = "Duration"
    for lan in language_list:
        entity_list += make_minutes(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
        entity_list += make_hours(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
        entity_list += make_seconds(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
    if amount is None:
        amount = len(entity_list)
    return random.choices(entity_list, k=amount)


def maker_RadioChannelPosition(language_list=['de', 'en'], amount=None, entity_amount=None, max_range=100,
                               random_seed=42):
    entity_list = []
    random.seed(random_seed)
    entity_type = "RadioChannelPosition"
    for lan in language_list:
        entity_list += make_positions(entity_type, lan, entity_amount=entity_amount, max_range=max_range,
                                      random_seed=random_seed)
    if amount is None:
        amount = len(entity_list)
    return random.choices(entity_list, k=amount)


def maker_TvChannelPosition(language_list=['de', 'en'], amount=None, entity_amount=None, max_range=100, random_seed=42):
    entity_list = []
    random.seed(random_seed)
    entity_type = "TvChannelPosition"
    for lan in language_list:
        entity_list += make_positions(entity_type, lan, entity_amount=entity_amount, max_range=max_range,
                                      random_seed=random_seed)
    if amount is None:
        amount = len(entity_list)
    return random.choices(entity_list, k=amount)


def maker_LocalsearchTimeStampStartTime(language_list=['de', 'en'], amount=None, entity_amount=None, random_seed=42):
    entity_list = []
    random.seed(random_seed)
    entity_type = "LocalsearchTimeStampStartTime"
    for lan in language_list:
        entity_list += make_timestamp_clock(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
    if amount is None:
        amount = len(entity_list)
    return random.choices(entity_list, k=amount)


def maker_LocalsearchTimeStampEndTime(language_list=['de', 'en'], amount=None, entity_amount=None, random_seed=42):
    entity_list = []
    random.seed(random_seed)
    entity_type = "LocalsearchTimeStampEndTime"
    for lan in language_list:
        entity_list += make_timestamp_clock(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
    if amount is None:
        amount = len(entity_list)
    return random.choices(entity_list, k=amount)


def maker_LocalsearchTimeStampStartDay(language_list=['de', 'en'], amount=None, entity_amount=None, random_seed=42):
    entity_list = []
    random.seed(random_seed)
    entity_type = "LocalsearchTimeStampStartDay"
    for lan in language_list:
        entity_list += make_timestamp_date(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
        entity_list += make_timestamp_word(entity_type, lan, entity_amount=entity_amount, random_seed=random_seed)
    if amount is None:
        amount = len(entity_list)
    return random.choices(entity_list, k=amount)
