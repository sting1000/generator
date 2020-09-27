from generator.tools import filter_aliases
from plato_ai_asr_preprocessor.preprocessor import Preprocessor
import random
import re


class Generator:
    def __init__(self, templates, entities):
        self.templates = templates
        self.entities = entities

    def get_entity(self, tag: str, target_lang: str):
        """
        randomly generate an entity given language and spcific tag
        return: a row of Dataframe
        """
        # check input existence
        if tag not in self.entities['type'].values:
            print('ERROR: Tag {} does not exist, try anther one.'.format(tag))
        if target_lang not in self.entities['language'].values:
            print('ERROR: Language {} does not exist, try anther one.'.format(target_lang))

        # random get one 
        selected_entities = self.entities[self.entities['type'] == tag][self.entities['language'] == target_lang]
        selected_entity = selected_entities.sample(n=1)
        return selected_entity

    def get_value_from_tag(self, tag: str, target_lang: str) -> str:
        """
        randomly generate a value according to tag (such as 'MyCloudArea') and language (like "fr")
        """
        # get entity
        selected_entity = self.get_entity(tag, target_lang)
        
        # find all values in the entity
        aliases = filter_aliases(selected_entity['aliases'].values[0])
        value = selected_entity['value'].values[0]
        normalized_value = selected_entity['normalizedValue'].values[0]

        #random get one
        selected_value = random.choice(aliases + [value, normalized_value])
        return selected_value

    def get_template(self, target_id: str, target_lang: str) -> str:
        """
        radomly choose a template from given id and language
        """
        pattern_list = self.templates[self.templates['id'] == target_id][target_lang].values[0]['texts']
        pattern = random.choice(pattern_list)['ttsText'] 
        
        return pattern

    def remove_tags(self, template: str, target_lang: str) -> str:
        """
        replace tags in template accordingly

        E.g. {MyCloudArea} auf #myCloud anzeigen -> Fotos auf #myCloud anzeigen
        """
        tags = [s[1 : -1] for s in re.findall(r'{\S+}', template)]

        for tag in tags:
            selected_value = self.get_value_from_tag(tag, target_lang)
            template = re.sub('{' + tag + '}', selected_value, template, 1)
            
        return template

    def get_command(self, target_id: str, target_lang: str, normalizer = None, verbose = False) -> str:
        """
        generate a command given id and language
        """
        
        template = self.get_template(target_id, target_lang)
        if verbose: 
            print("Choose template: \n\t{}".format(template)) 

        template = self.remove_tags(template, target_lang)
        if verbose: 
            print("After tag removal: \n\t{}".format(template)) 

        template  = normalizer(template, target_lang) if normalizer else template
        if verbose:
            print("After normalizer: \n\t{}".format(template)) 
            
        return template


class Normalizer:
    preprocessor: Preprocessor = None

    def __init__(self):
        self.preprocessor = Preprocessor(use_case='kaldi-lm', cleaner_config=None, abbreviation_config=None)

    def normalize_text(self, text: str, language: str) -> str:
        normalized_text, _ = self.preprocessor.process(text=text, language=language)
        return normalized_text