from plato_ai_asr_preprocessor.preprocessor import Preprocessor


class Processor:
    preprocessor: Preprocessor = None

    def __init__(self):
        self.preprocessor = Preprocessor(use_case='kaldi-lm', cleaner_config=None, abbreviation_config=None)

    def normalize_text(self, text: str, language: str) -> str:
        normalized_text, _ = self.preprocessor.process(text=text, language=language)
        return normalized_text
