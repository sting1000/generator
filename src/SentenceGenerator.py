import re
import os
import errno
import random
import itertools
from collections import OrderedDict
from tqdm import tqdm
from utils import Normalizer
import warnings

warnings.filterwarnings("ignore")


def check_filename(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


class SentenceGenerator:
    def __init__(self, templates, entities, LRU_size=100, max_holder_amount=2, max_combo_amount=3000):
        self.tagging = 0
        self.padding = 0
        self.templates = templates
        self.entities = entities
        self.normalizer = Normalizer().normalize_text
        self.tags_entities_dic = LRUCache(size=LRU_size)
        self.max_combo_amount = max_combo_amount
        self.sentence_num = 0
        self.max_holder_amount = max_holder_amount

    def permute(self, output_file, tagging=0, padding=0):
        print("Start Permutation...")
        check_filename(output_file)
        self.tagging = tagging
        self.padding = padding

        with open(output_file, 'w') as outfile:
            self.__print_header(outfile)
            for row_index, row in tqdm(self.templates.iterrows()):
                language = row['language']
                entity_holder = re.findall(r'{\S+}', row['text'])
                if entity_holder:
                    if len(entity_holder) <= self.max_holder_amount:
                        entities_combo_list = self.__combine_entities(entity_holder, language=language)
                        for entities_combo in entities_combo_list:
                            self.__print_pad(outfile, language=language, value='<sos>')
                            self.__print_combo(outfile, row, entity_holder, entities_combo)
                            self.__print_pad(outfile, language=language, value='<eos>')
                            self.sentence_num += 1
                else:
                    self.__print_pad(outfile, language=language, value='<sos>')
                    self.__print_plain(outfile, row)
                    self.__print_pad(outfile, language=language, value='<eos>')
                    self.sentence_num += 1

    def __combine_entities(self, tags, language):
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

        if len(all_entities_combo) > self.max_combo_amount:
            all_entities_combo = random.sample(all_entities_combo, self.max_combo_amount)
        return all_entities_combo

    def __print_pad(self, outfile, language, value):
        for _ in range(self.padding):
            pad = self.__get_item_pad(language, value)
            self.__print_item(outfile, pad)

    def __print_header(self, outfile):
        # print dict item to outfile with certain order
        columns = ['sentence_id', 'language', 'intent', 'written', 'spoken', 'type']
        if self.tagging:
            columns += ['token_id', 'token', 'tag']
        outfile.write(','.join(columns) + '\n')

    def __print_item(self, outfile, item):
        # print dict item to outfile with certain order
        columns = ['sentence_id', 'language', 'intent', 'written', 'spoken', 'type']
        if self.tagging:
            columns += ['token_id', 'token', 'tag']
        value_list = []
        for col in columns:
            value_list.append(item[col])
        outfile.write(','.join(value_list) + '\n')

    def __get_item_pad(self, language, value):
        item = {
            'sentence_id': str(self.sentence_num),
            'intent': 'Pad',
            'language': language,
            'written': value,
            'spoken': value,
            'type': 'Pad'
        }
        if self.tagging:
            item['token_id'] = '-1'
            item['token'] = value
            item['tag'] = 'O'
        return item

    def __print_combo(self, outfile, row, entity_holder, entities_combo):
        item = {
            'sentence_id': str(self.sentence_num),
            'intent': row['intent'],
            'language': row['language']
        }

        holder_index = 0
        for token_index, token in enumerate(row['text'].split()):
            if re.findall(r'{\S+}', token):
                item['written'] = entities_combo[holder_index]
                item['spoken'] = self.normalizer(item['written'], item['language'])
                item['type'] = entity_holder[holder_index][1:-1]
                holder_index += 1
            else:
                token = str(token)
                item['written'] = token
                item['spoken'] = token
                item['type'] = 'plain'

            # For changed case
            if item['type'] != 'plain' and item['written'] != item['spoken']:
                for i, word in enumerate(item['spoken'].split()):
                    item['token'] = word
                    item['token_id'] = str(token_index)
                    if i == 0:
                        item['tag'] = 'B-TBNorm'
                    else:
                        item['tag'] = 'I-TBNorm'
                    self.__print_item(outfile, item)
            else:
                if self.tagging:
                    for word in item['spoken'].split():
                        item['token'] = str(word)
                        item['token_id'] = str(token_index)
                        item['tag'] = 'O'
                        self.__print_item(outfile, item)
                else:
                    self.__print_item(outfile, item)

    def __print_plain(self, outfile, row):
        item = {
            'sentence_id': str(self.sentence_num),
            'intent': row['intent'],
            'language': row['language'],
            'type': 'plain',
            'tag': 'O'
        }
        for token_index, token in enumerate(row['text'].split()):
            token = str(token)
            item['written'] = token
            item['spoken'] = token
            item['token'] = token
            item['token_id'] = str(token_index)
            self.__print_item(outfile, item)


class LRUCache:
    def __init__(self, size):
        self.size = size
        self.linked_map = OrderedDict()

    def set(self, key, value):
        if key in self.linked_map:
            self.linked_map.pop(key)

        if self.size == len(self.linked_map):
            self.linked_map.popitem(last=False)
        self.linked_map.update({key: value})

    def get(self, key):
        if key in self.linked_map:
            value = self.linked_map.get(key)
            self.linked_map.pop(key)
            self.linked_map.update({key: value})
            return value
        else:
            return None
