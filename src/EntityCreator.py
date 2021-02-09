from src.entity_helper import make_hour, make_minute, make_second, make_position
from src.entity_helper import make_timestamp_date, make_timestamp_clock, make_timestamp_word


class EntityCreator:
    def __init__(self, language, max_position_range):
        self.language = language
        self.max_range = max_position_range

    def generate_RouterWiFiDuration(self):
        entity_list = []
        entity_list += make_minute("RouterWiFiDuration", self.language)
        entity_list += make_hour("RouterWiFiDuration", self.language)
        return entity_list

    def generate_Duration(self):
        entity_list = []
        entity_type = "Duration"
        entity_list += make_second(entity_type, self.language)
        entity_list += make_minute(entity_type, self.language)
        entity_list += make_hour(entity_type, self.language)
        return entity_list

    def generate_RadioChannelPosition(self):
        entity_list = []
        entity_type = "RadioChannelPosition"
        entity_list += make_position(entity_type, self.language, max_range=self.max_range)
        return entity_list

    def generate_TvChannelPosition(self):
        entity_list = []
        entity_type = "TvChannelPosition"
        entity_list += make_position(entity_type, self.language, max_range=self.max_range)
        return entity_list

    def generate_LocalsearchTimeStampStartTime(self):
        entity_list = []
        entity_type = "LocalsearchTimeStampStartTime"
        entity_list += make_timestamp_clock(entity_type, self.language)
        return entity_list

    def generate_LocalsearchTimeStampEndTime(self):
        entity_list = []
        entity_type = "LocalsearchTimeStampEndTime"
        entity_list += make_timestamp_clock(entity_type, self.language)
        return entity_list

    def generate_LocalsearchTimeStampStartDay(self):
        entity_list = []
        entity_type = "LocalsearchTimeStampStartDay"
        entity_list += make_timestamp_date(entity_type, self.language)
        entity_list += make_timestamp_word(entity_type, self.language)
        return entity_list
