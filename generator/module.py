import pandas as pd
from generator.tools import filter_aliases, remove_sharp_sign, assign_tag_to_words, make_bio_tag
from generator.normalizer import Normalizer
from generator.cleaning import normalizeString
import random
import re
import itertools
import time
from tqdm import tqdm


class Generator:
    def __init__(self, templates, entities, method='all', name=None):
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
        self.command_pool = {}
        self.tag_dic = {}
        self.template_dic = {}
        self.name = name
        self.tag_tuples_dic = {}

    def get_values_from_tag(self, tag: str, target_lang: str) -> list:
        """
        get value(s) according to class method (one/ all) given a specific tag

        Args:
            tag: string like MyCloudArea
            target_lang: fr de en it

        Returns: list of possible value(s)

        """
        if tag not in self.tag_dic:
            # TODO: comment language selection to have more permutation
            selected_entities = self.entities[self.entities['type'] == tag][self.entities['language'] == target_lang]

            filtered_values = []
            # find all values in the entity
            for _, selected_entity in selected_entities.iterrows():
                aliases = selected_entity['aliases']
                value = [str(selected_entity['value'])]
                normalized_value = [selected_entity['normalizedValue']]
                filtered_values += filter_aliases(aliases + value + normalized_value, target_lang)
            self.tag_dic[tag] = filtered_values
        selected_values = self.apply_method(self.tag_dic[tag])
        return selected_values

    def get_templates(self, target_id: str, target_lang: str) -> list:
        """
        get template(s) according to class method (one/all)
        Args:
            target_id: the init id in template dataframe
            target_lang: language of the template

        Returns: list of template(s)

        """
        if target_id not in self.template_dic:
            pattern_list = self.templates[self.templates['id'] == target_id][target_lang].values[0]['texts']
            pattern_list = [p['ttsText'] for p in pattern_list]
            self.template_dic['target_id'] = pattern_list
        return self.apply_method(self.template_dic['target_id'])

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
            template = template.split()  # [token.strip('#') for token in template.split()]
            template_command_pool = []

            def replace_pos(pos: int, curr_temp: list, curr_label: list):
                if pos == len(template):
                    template_command_pool.append((curr_temp, curr_label))
                    return

                if re.findall(r'{\S+}', template[pos]):
                    tag = template[pos][1: -1]
                    for selected_value in self.get_values_from_tag(tag, target_lang):
                        selected_value = selected_value.split()
                        for token in selected_value:
                            curr_label.extend(self.normalizer(token, target_lang).split())
                        curr_temp.extend(selected_value)
                        replace_pos(pos + 1, curr_temp, curr_label)
                else:
                    curr_temp.append(template[pos])
                    curr_label.append(template[pos])
                    replace_pos(pos + 1, curr_temp, curr_label)

            replace_pos(0, [], [])
            # self.command_pool.update(template_command_pool)
        if verbose:
            print("After tag removal: \n\t{}".format(self.command_pool))

        template_command_pool = [(normalizeString(' '.join(pair[0])), normalizeString(' '.join(pair[1]))) for pair in
                                 template_command_pool]
        return template_command_pool  # self.command_pool

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

    def permute(self, threshold=3000) -> pd.DataFrame:
        print("Making permutation for " + self.name)
        time_start = time.time()
        df_pool = pd.DataFrame(columns=["id", "language", "spoken", "written", "entities_dic"])
        for _, row in tqdm(self.templates.iterrows()):
            tags = re.findall(r'{\S+}', row['text'])
            if tags:
                if ''.join(tags) in self.tag_tuples_dic:
                    tag_tuples = self.tag_tuples_dic[''.join(tags)]
                else:
                    meta_tag_list = []
                    for tag in tags:
                        type_condition = self.entities['type'] == tag[1: -1]
                        lan_condition = self.entities['language'] == row['language']
                        val_list = self.entities[type_condition][lan_condition]['value'].values
                        meta_tag_list.append(val_list)
                    tag_tuples = list(itertools.product(*meta_tag_list))
                    self.tag_tuples_dic[''.join(tags)] = tag_tuples

                if len(tag_tuples) > threshold:
                    tag_tuples = random.sample(tag_tuples, threshold)
                for tup in tag_tuples:
                    text = row['text']
                    entities_dic = {}
                    for ind in range(len(tup)):
                        text = re.sub(tags[ind], tup[ind], text, count=1)
                        entities_dic[normalizeString(tup[ind])] = tags[ind]
                    written = text
                    spoken = self.normalizer(written, row['language'])
                    df_pool = df_pool.append({
                        "id": row['id'],
                        "language": row['language'],
                        "spoken": normalizeString(spoken),
                        "written": normalizeString(written),
                        "entities_dic": entities_dic
                    }, ignore_index=True)
            else:
                written = row['text']
                spoken = self.normalizer(written, row['language'])
                df_pool = df_pool.append({
                         "id": row['id'],
                         "language": row['language'],
                         "spoken":  normalizeString(spoken),
                         "written": normalizeString(written),
                         "entities_dic": {}
                          }, ignore_index=True)

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        return df_pool
