from classes.command_generator.cleaning import clean_string


class Command:
    def __init__(self, intent=None, language=None, spoken=None, written=None, entities_dic=None):
        self.intent = intent
        self.language = language
        self.spoken = clean_string(spoken) if spoken else None
        self.written = clean_string(written) if written else None
        self.entities_dic = entities_dic

    def get_json(self):
        data = {
            "id": self.intent,
            "language": self.language,
            "spoken": self.spoken,
            "written": self.written,
            "entities_dic": self.entities_dic
        }
        return data

    def set_id_(self, id_):
        self.intent = id_

    def set_spoken(self, spoken):
        self.spoken = clean_string(spoken)

    def set_spoken_from_written(self, normalizer):
        spoken = normalizer(self.written, self.language)
        self.spoken = clean_string(spoken)

    def set_written(self, written):
        self.written = clean_string(written)

    def set_language(self, language):
        self.language = language

    def set_entities_dic(self, entities_dic):
        self.entities_dic = entities_dic
