from generator.tools import filter_aliases, remove_sharp_sign, assign_tag_to_words, make_bio_tag
from generator.normalizer import Normalizer
from tqdm import tqdm
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
        self.command_pool = []

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

        # TODO: comment language selection to have more permutation
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

    def get_command(self, target_id: str, target_lang: str, verbose=False) -> list:
        """
        generate a command given id and language
        Args:
            size:
            target_id: the init id in template dataframe
            target_lang: language of template
            verbose: True to activate print and check the middle state

        Returns: list of tuples: [(template, label)]

        """
        templates = self.get_templates(target_id, target_lang)
        if verbose:
            print("Choose template: \n\t{}".format(templates))

        for template in templates:
            label = []
            template = [token.strip('#') for token in template.split()]
            template_command_pool = []

            def replace_pos(pos: int, curr_temp: list, curr_label: list):
                if pos == len(template):
                    template_command_pool.append((curr_temp, curr_label))
                    return

                if re.findall(r'{\S+}', template[pos]):
                    tag = template[pos][1: -1]
                    for selected_value in self.get_values_from_tag(tag, target_lang):
                        selected_value_norm = self.normalizer(selected_value, target_lang).split()
                        curr_temp.extend(selected_value_norm)
                        curr_label.extend(make_bio_tag(selected_value_norm, tag))
                        replace_pos(pos + 1, curr_temp, curr_label)
                else:
                    curr_temp.append(template[pos])
                    curr_label.append('O')
                    replace_pos(pos + 1, curr_temp, curr_label)

            replace_pos(0, [], [])
            self.command_pool += template_command_pool
        if verbose:
            print("After tag removal: \n\t{}".format(self.command_pool))
        return template_command_pool#self.command_pool

    def apply_method(self, li: list) -> list:
        """
        To use self.method choose from li
        """
        if self.method == "one":
            selected_values = [random.choice(li)]
        elif self.method == "all":
            selected_values = li
        else:
            print('ERROR: Method {} does not exist, try anther one.'.format(self.method))
            selected_values = []
        return selected_values
