from generator.tools import filter_aliases
from generator.normalizer import Normalizer
import random
import re

class Generator:
    def __init__(self, templates, entities, method):
        """
        method: "one", "all"
        """
        self.templates = templates
        self.entities = entities
        self.normalizer = Normalizer().normalize_text
        self.method = method

    def get_values_from_tag(self, tag: str, target_lang: str) -> list:
        """
        randomly generate a value according to tag (such as 'MyCloudArea') and language (like "fr")
        """
        # get entity
        if tag not in self.entities['type'].values:
            print('ERROR: Tag {} does not exist, try anther one.'.format(tag))
            return
        if target_lang not in self.entities['language'].values:
            print('ERROR: Language {} does not exist, try anther one.'.format(target_lang))
            return
        selected_entities = self.entities[self.entities['type'] == tag][self.entities['language'] == target_lang]
        
        filtered_values = []
        # find all values in the entity
        for _, selected_entity in selected_entities.iterrows():
            aliases = selected_entity['aliases']
            value = [str(selected_entity['value'])]
            normalized_value = [selected_entity['normalizedValue']]
            filtered_values += filter_aliases(aliases + value + normalized_value, target_lang)

        selected_values = self.apply_method(filtered_values)
        return selected_values

    def get_templates(self, target_id: str, target_lang: str) -> list:
        """
        radomly choose a template from given id and language
        """
        pattern_list = self.templates[self.templates['id'] == target_id][target_lang].values[0]['texts']
        return self.apply_method([p['ttsText'] for p in pattern_list])

    def replace_tags(self, template: str, target_lang: str) -> list:
        """
        replace tags in template accordingly

        E.g. {MyCloudArea} auf #myCloud anzeigen -> Fotos auf #myCloud anzeigen
        """
        template_command_pool = []
        tags = [s[1 : -1] for s in re.findall(r'{\S+}', template)]
        
        if not tags:
            return [template]
        for tag in tags:
            selected_values = self.get_values_from_tag(tag, target_lang)
            for selected_value in selected_values:
                template_command_pool.append(re.sub('{' + tag + '}', selected_value, template, 1))
        return template_command_pool

    def get_command(self, target_id: str, target_lang: str, verbose = False) -> list:
        """
        generate a command given id and language
        """
        command_pool = []
        templates = self.get_templates(target_id, target_lang)
        if verbose: 
            print("Choose template: \n\t{}".format(templates)) 
        
        for template in templates:
            template = self.remove_sharp_sign(template)
            command_pool += self.replace_tags(template, target_lang)
        if verbose: 
            print("After tag removal: \n\t{}".format(command_pool)) 

        return command_pool

    def apply_method(self, li:list):
        if self.method == "one":
            selected_values = [random.choice(li)]
        elif self.method == "all":
            selected_values = li
        else:
            print('ERROR: Method {} does not exist, try anther one.'.format(self.method))
        return selected_values
    
    def remove_sharp_sign(self, sentence: str) -> str:
        search = re.search(r'\s#[a-zA-Z]+', sentence)
        while search:
            pos = search.start() + 1
            sentence = sentence[:pos] + sentence[(pos+1):]
            search = re.search(r'\s#[a-zA-Z]+', sentence)
        return sentence