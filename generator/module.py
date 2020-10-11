from generator.tools import filter_aliases, remove_sharp_sign, assign_tag_to_words
from generator.normalizer import Normalizer
import random
import re

class Generator:
    def __init__(self, templates, entities, method):
        """
        init function
        Args:
            templates: Dataframe of templates
            entities: Dataframe of entities
            method: "one"/ "all", one to random choice one template, all to generate all possible templates
        """
        self.templates = templates
        self.entities = entities
        self.normalizer = Normalizer().normalize_text
        self.method = method

    def get_values_from_tag(self, tag: str, target_lang: str) -> list:
        """
        get value(s) according to class method (one/ all) given a specific tag

        Args:
            tag: string like MyCloudArea
            target_lang: fr de en it

        Returns: list of possible value(s)

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
        get template(s) according to class method (one/all)
        Args:
            target_id: the init id in template dataframe
            target_lang: language of the template

        Returns: list of template(s)

        """
        pattern_list = self.templates[self.templates['id'] == target_id][target_lang].values[0]['texts']
        return self.apply_method([p['ttsText'] for p in pattern_list])

    def replace_tags(self, template: str, label: str, target_lang: str) -> list:
        """
        replace tags in template accordingly, and given string of label

        E.g. {MyCloudArea} auf myCloud anzeigen -> Fotos auf myCloud anzeigen, {MyCloudArea} {Template} {Template} {
        Template}

        Args:
            template: string of template
            label: string of labels like {MyCloudArea}, {Template}
            target_lang: language of template

        Returns: list of tuples: [(template, label)]

        """
        template_command_pool = []
        tags = [s[1: -1] for s in re.findall(r'{\S+}', template)]

        if not tags:
            return [(template, label)]
        for tag in tags:
            selected_values = self.get_values_from_tag(tag, target_lang)
            for selected_value in selected_values:
                replaced_template = re.sub('{' + tag + '}', selected_value, template, 1)
                replaced_label = re.sub('{' + tag + '}', selected_value, label, 1)
                replaced_label = assign_tag_to_words(replaced_label, '{' + tag + '}')
                template_command_pool += self.replace_tags(replaced_template, replaced_label, target_lang)
        return self.apply_method(template_command_pool)

    def get_command(self, target_id: str, target_lang: str, verbose=False) -> list:
        """
        generate a command given id and language
        Args:
            target_id: the init id in template dataframe
            target_lang: language of template
            verbose: True to activate print and check the middle state

        Returns: list of tuples: [(template, label)]

        """
        command_pool = []
        templates = self.get_templates(target_id, target_lang)
        if verbose:
            print("Choose template: \n\t{}".format(templates))

        for template in templates:
            template = remove_sharp_sign(template)
            label = assign_tag_to_words(template, "{Template}")
            command_pool += self.replace_tags(template, label, target_lang)
        if verbose:
            print("After tag removal: \n\t{}".format(command_pool))

        return command_pool

    def apply_method(self, li: list) -> list:
        """
        To use self.method choose from li
        Args:
            li: list of values

        Returns: list of values

        """
        if self.method == "one":
            selected_values = [random.choice(li)]
        elif self.method == "all":
            selected_values = li
        else:
            print('ERROR: Method {} does not exist, try anther one.'.format(self.method))
            selected_values = []
        return selected_values
