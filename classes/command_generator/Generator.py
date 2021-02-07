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
            print("Making permutation for " + self.name)
            for ind_row, row in tqdm(self.templates.iterrows()):
                tags = re.findall(r'{\S+}', row['text'])
                if tags:
                    # if have entity
                    if len(tags) <= self.max_tag_amount:
                        entities_combo_list = self.get_entities_combo(tags, language=row['language'], random_seed=ind_row)
                        for entities_combo in entities_combo_list:
                            # for each kind of entities combination
                            if pad:
                                self.print_pad(outfile, tag, pad_size)

                            comb_id = 0
                            token_id = 1
                            for token in row['text'].split():
                                item = {
                                    'sentence_id': str(self.sentence_num),
                                    'intent': row['intent'],
                                    'language': row['language']
                                }
                                if token[0] == '{' and token[-1] == '}':
                                    item['written'] = entities_combo[comb_id]
                                    item['spoken'] = self.normalizer(item['written'], item['language'])
                                    item['type'] = tags[comb_id][1:-1]
                                    comb_id += 1
                                else:
                                    item['written'] = token
                                    item['spoken'] = token
                                    item['type'] = 'plain'

                                if tag:
                                    if item['type'] != 'plain' and item['written'] != item['spoken']:
                                        for i, word in enumerate(item['spoken'].split()):
                                            item['token'] = word
                                            item['token_id'] = str(token_id)
                                            if i == 0:
                                                item['tag'] = 'B-TBNorm'
                                            else:
                                                item['tag'] = 'I-TBNorm'
                                            outfile.write(','.join(item.values()) + '\n')
                                    else:
                                        for word in item['spoken'].split():
                                            item['token'] = word
                                            item['token_id'] = str(token_id)
                                            item['tag'] = 'O'
                                            outfile.write(','.join(item.values()) + '\n')

                                else:
                                    outfile.write(','.join(item.values()) + '\n')
                                token_id += 1

                            if pad:
                                self.print_pad(outfile, tag, pad_size)
                            self.sentence_num += 1

                else:
                    # if no entity
                    if pad:
                        self.print_pad(outfile, tag, pad_size)

                    token_id = 1
                    for token in row['text'].split():
                        item = {
                            'sentence_id': str(self.sentence_num),
                            'intent': row['intent'],
                            'language': row['language'],
                            'written': token,
                            'spoken': token,
                            'type': 'plain',
                        }
                        if tag:
                            for word in item['spoken'].split():
                                item['token'] = word
                                item['token_id'] = str(token_id)
                                item['tag'] = 'O'
                                outfile.write(','.join(item.values()) + '\n')
                        else:
                            outfile.write(','.join(item.values()) + '\n')
                        token_id += 1

                    if pad:
                        self.print_pad(outfile, tag, pad_size)
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

    def print_pad(self, outfile, tag, pad_size):
        item = {
            'sentence_id': str(self.sentence_num),
            'intent': 'Pad',
            'language': '',
            'written': '',
            'spoken': '',
            'type': 'plain'
        }
        if tag:
            item['token_id'] = ''
            item['token'] = ''
            item['tag'] = 'O'
        for i in range(pad_size):
            outfile.write(','.join(item.values()) + '\n')