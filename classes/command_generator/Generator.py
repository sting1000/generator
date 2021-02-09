import re
import os
import errno
import random
import itertools
from tqdm import tqdm
from classes.command_generator.Normalizer import Normalizer
from classes.command_generator.LRUCache import LRUCache
import warnings
warnings.filterwarnings("ignore")


def check_filename(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


class Generator:
    def __init__(self, templates, entities, LRU_size=100, max_holder_amount=2, max_combo_amount=3000):
        self.has_no_output = True
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
        with open(output_file, 'w') as outfile:
            self.__print_header(outfile, tagging)
            for row_index, row in tqdm(self.templates.iterrows()):
                language = row['language']
                entity_holder = re.findall(r'{\S+}', row['text'])
                if entity_holder:
                    if len(entity_holder) <= self.max_holder_amount:
                        entities_combo_list = self.__combine_entities(entity_holder, language=language,
                                                                      random_seed=row_index)
                        for entities_combo in entities_combo_list:
                            self.__print_pad(outfile, tagging, padding, language=language, value='<sos>')
                            self.__print_combo(outfile, tagging, row, entity_holder, entities_combo)
                            self.__print_pad(outfile, tagging, padding, language=language, value='<eos>')
                            self.sentence_num += 1
                else:
                    self.__print_pad(outfile, tagging, padding, language=language, value='<sos>')
                    self.__print_plain(outfile, tagging, row)
                    self.__print_pad(outfile, tagging, padding, language=language, value='<eos>')
                    self.sentence_num += 1

    def __combine_entities(self, tags, language, random_seed):
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
            random.seed(random_seed)
            all_entities_combo = random.sample(all_entities_combo, self.max_combo_amount)
        return all_entities_combo

    def __get_item_pad(self, tagging, language, value):
        item = {
            'sentence_id': str(self.sentence_num),
            'intent': 'Pad',
            'language': language,
            'written': value,
            'spoken': value,
            'type': 'Pad'
        }
        if tagging:
            item['token_id'] = '-1'
            item['token'] = value
            item['tag'] = 'O'
        return item

    def __print_combo(self, outfile, tagging, row, entity_holder, entities_combo):
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
                item['written'] = token
                item['spoken'] = token
                item['type'] = 'plain'

            if item['type'] != 'plain' and item['written'] != item['spoken']:
                for i, word in enumerate(item['spoken'].split()):
                    item['token'] = word
                    item['token_id'] = str(token_index)
                    if i == 0:
                        item['tag'] = 'B-TBNorm'
                    else:
                        item['tag'] = 'I-TBNorm'
                    self.__print_item(outfile, item, tagging)
            else:
                for word in item['spoken'].split():
                    item['token'] = word
                    item['token_id'] = str(token_index)
                    item['tag'] = 'O'
                    self.__print_item(outfile, item, tagging)

    def __print_plain(self, outfile, tagging, row):
        item = {
            'sentence_id': str(self.sentence_num),
            'intent': row['intent'],
            'language': row['language'],
            'type': 'plain',
            'tag': 'O'
        }
        for token_index, token in enumerate(row['text'].split()):
            item['written'] = token
            item['spoken'] = token
            item['token'] = token
            item['token_id'] = str(token_index)
            self.__print_item(outfile, item, tagging)

    def __print_pad(self, outfile, tagging, padding, language, value):
        for _ in range(padding):
            pad = self.__get_item_pad(tagging, language, value)
            self.__print_item(outfile, pad, tagging)

    def __print_item(self, outfile, item, tagging):
        # print dict item to outfile with certain order
        columns = ['sentence_id', 'language', 'intent', 'written', 'spoken', 'type']
        if tagging:
            columns += ['token_id', 'token', 'tag']
        value_list = []
        for col in columns:
            value_list.append(item[col])
        outfile.write(','.join(value_list) + '\n')

    def __print_header(self, outfile, tagging):
        # print dict item to outfile with certain order
        columns = ['sentence_id', 'language', 'intent', 'written', 'spoken', 'type']
        if tagging:
            columns += ['token_id', 'token', 'tag']
        outfile.write(','.join(columns) + '\n')
