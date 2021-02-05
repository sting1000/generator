import json
from classes.command_generator.Command import Command
from classes.command_generator.LRUCache import LRUCache
import random
import re
import itertools
from tqdm import tqdm
import pandas as pd


class Generator:
    def __init__(self, templates, entities, normalizer, name=None, threshold=3000, LRU_size=100, max_tag_amount=2):
        self.has_no_output = True
        self.templates = templates
        self.entities = entities
        self.normalizer = normalizer
        self.name = name
        self.tags_entities_dic = LRUCache(size=LRU_size)
        self.threshold = threshold
        self.sentence_num = 0
        self.max_tag_amount = max_tag_amount

    def permute(self, output_path, tag=False, pad=False, pad_size=1):
        # init
        with open(output_path / (self.name + '.csv'), 'w') as outfile:
            # columns = ['sentence_id', 'intent', 'language', 'written', 'spoken', 'type']
            # outfile.write(','.join(columns) + '\n')

            print("Making permutation for " + self.name)
            for ind_row, row in tqdm(self.templates.iterrows()):
                tags = re.findall(r'{\S+}', row['text'])
                if tags and len(tags) <= self.max_tag_amount:
                    entities_combo_list = self.get_entities_combo(tags, language=row['language'], random_seed=ind_row)
                    for entities_combo in entities_combo_list:
                        ind = 0
                        if pad:
                            item = {
                                'sentence_id': str(self.sentence_num),
                                'intent': 'Pad',
                                'language': row['language'],
                                'written': '<sep>',
                                'spoken': '<sep>',
                                'type': 'plain'
                            }
                            if tag:
                                item['token'] = '<sep>'
                                item['tag'] = 'O'
                            for i in range(pad_size):
                                outfile.write(','.join(item.values()) + '\n')

                        for token in row['text'].split():
                            item = {
                                'sentence_id': str(self.sentence_num),
                                'intent': row['intent'],
                                'language': row['language']
                            }
                            if token[0] == '{' and token[-1] == '}':
                                item['written'] = entities_combo[ind]
                                item['spoken'] = self.normalizer(item['written'], item['language'])
                                item['type'] = tags[ind][1:-1]
                                ind += 1
                            else:
                                item['written'] = token
                                item['spoken'] = token
                                item['type'] = 'plain'

                            if tag:
                                if item['type'] != 'PLAIN' and item['written'] != item['spoken']:
                                    for i, word in enumerate(item['spoken'].split()):
                                        item['token'] = word
                                        if i == 0:
                                            item['tag'] = 'B-TBNorm'
                                        else:
                                            item['tag'] = 'I-TBNorm'
                                        outfile.write(','.join(item.values()) + '\n')
                                else:
                                    for _, word in enumerate(item['spoken'].split()):
                                        item['token'] = word
                                        item['tag'] = 'O'
                                        outfile.write(','.join(item.values()) + '\n')
                            else:
                                outfile.write(','.join(item.values()) + '\n')
                        self.sentence_num += 1

    def get_entities_combo(self, tags, language, random_seed):
        key = language + '_'.join(tags)
        all_entities_combo = self.tags_entities_dic.get(key)
        if not all_entities_combo:
            tags_value_list = []
            for tag in tags:
                type_condition = self.entities['type'] == tag[1: -1]
                language_condition = self.entities['language'] == language
                val_list = self.entities[type_condition][language_condition]['value'].values
                tags_value_list.append(val_list)
            all_entities_combo = list(itertools.product(*tags_value_list))
            self.tags_entities_dic.set(key, all_entities_combo)

        if len(all_entities_combo) > self.threshold:
            random.seed(random_seed)
            all_entities_combo = random.sample(all_entities_combo, self.threshold)
        return all_entities_combo
