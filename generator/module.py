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

    def get_value_from_tag(self, tag: str, target_lang: str) -> list:
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
        
        # find all values in the entity
        values_pool = []
        for selected_entity in selected_entities:
            aliases = selected_entity['aliases'].values[0]
            value = [selected_entity['value'].values[0]]
            normalized_value = [selected_entity['normalizedValue'].values[0]]
            filtered_values = filter_aliases(aliases + value + normalized_value)

            for value in filtered_values:
                filtered_values.remove(value)
                normalized_value = self.normalizer(value, target_lang)
                if normalized_value in filtered_values:
                    filtered_values.remove(normalized_value)
                filtered_values.append(value)
            values_pool += filtered_values
        
        selected_values = self.apply_method(filtered_values)
        return selected_values

    def get_template(self, target_id: str, target_lang: str) -> str:
        """
        radomly choose a template from given id and language
        """
        pattern_list = self.templates[self.templates['id'] == target_id][target_lang].values[0]['texts']
        pattern = random.choice(pattern_list)['ttsText'] 
        
        return pattern

    def remove_tags(self, template: str, target_lang: str) -> list:
        """
        replace tags in template accordingly

        E.g. {MyCloudArea} auf #myCloud anzeigen -> Fotos auf #myCloud anzeigen
        """
        template_command_pool = []
        tags = [s[1 : -1] for s in re.findall(r'{\S+}', template)]
        for tag in tags:
            selected_values = self.get_value_from_tag(tag, target_lang)
            for selected_value in self.apply_method(selected_values):
                template_command_pool.append(re.sub('{' + tag + '}', selected_value, template, 1))
        return template_command_pool

    def get_command(self, target_id: str, target_lang: str, verbose = False) -> str:
        """
        generate a command given id and language
        """
        
        template = self.get_template(target_id, target_lang)
        if verbose: 
            print("Choose template: \n\t{}".format(template)) 

        template = self.remove_tags(template, target_lang)
        if verbose: 
            print("After tag removal: \n\t{}".format(template)) 

        # template  = self.normalizer(template, target_lang)
        # if verbose:
        #     print("After normalizer: \n\t{}".format(template)) 

        return template

    def apply_method(self, li:list):
        if self.method == "one":
            selected_values = [random.choice(li.sample)]
        elif self.method == "all":
            selected_values = li
        else:
            print('ERROR: Method {} does not exist, try anther one.'.format(self.method))
        return selected_values