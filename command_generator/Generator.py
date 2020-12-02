import json
from command_generator.Command import Command
from command_generator.cleaning import clean_string
from command_generator.LRUCache import LRUCache
import random
import re
import itertools
from tqdm import tqdm


def replace_tags_by_entities(text, tags, entities):
    entities_dic = {}
    for ind in range(len(entities)):
        entity = clean_string(entities[ind])
        text = re.sub(tags[ind], entity, text, count=1)
        entities_dic[entity] = tags[ind]
    return text, entities_dic


class Generator:
    def __init__(self, templates, entities, normalizer, name=None, threshold=3000, LRU_size=50, max_tag_amount=3):
        self.has_no_output = True
        self.templates = templates
        self.entities = entities
        self.normalizer = normalizer
        self.name = name
        self.tags_entities_dic = LRUCache(size=LRU_size)
        self.threshold = threshold
        self.current_command = Command()
        self.max_tag_amount = max_tag_amount

    def permute(self, output_path):
        print("Making permutation for " + self.name)
        with open(output_path, 'w') as outfile:
            outfile.write("[")
            for ind_row, row in tqdm(self.templates.iterrows()):
                self.current_command.set_id_(row['id'])
                self.current_command.set_language(row['language'])

                tags = re.findall(r'{\S+}', row['text'])
                if tags and len(tags) <= self.max_tag_amount:
                    entities_combo_list = self.get_entities_combo(tags, random_seed=ind_row)
                    for entities_combo in entities_combo_list:
                        written, entities_dic = replace_tags_by_entities(text=row['text'], tags=tags, entities=entities_combo)
                        self.current_command.set_written(written)
                        self.current_command.set_spoken_from_written(self.normalizer)
                        self.current_command.set_entities_dic(entities_dic)
                        self.dump_json(outfile, data=self.current_command.get_json())
                else:
                    self.current_command.set_written(row['text'])
                    self.current_command.set_spoken_from_written(self.normalizer)
                    self.current_command.set_entities_dic({})
                    self.dump_json(outfile, data=self.current_command.get_json())
            outfile.write("]")
        outfile.close()

    def get_entities_combo(self, tags, random_seed):
        tags_key = self.current_command.language + '_'.join(tags)
        all_entities_combo = self.tags_entities_dic.get(tags_key)
        if not all_entities_combo:
            tags_value_list = []
            for tag in tags:
                type_condition = self.entities['type'] == tag[1: -1]
                language_condition = self.entities['language'] == self.current_command.language
                val_list = self.entities[type_condition][language_condition]['value'].values
                tags_value_list.append(val_list)
            all_entities_combo = list(itertools.product(*tags_value_list))
            self.tags_entities_dic.set(tags_key, all_entities_combo)

        if len(all_entities_combo) > self.threshold:
            random.seed(random_seed)
            all_entities_combo = random.sample(all_entities_combo, self.threshold)
        return all_entities_combo

    def dump_json(self, outfile, data):
        if self.has_no_output:
            outfile.write(json.dumps(data))
        else:
            outfile.write(",")
            outfile.write(json.dumps(data))
        self.has_no_output = False
