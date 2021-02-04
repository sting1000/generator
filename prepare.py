import time
from classes.entity_maker.helper import merge_entity_types
import pandas as pd
from pathlib import Path
import json
from classes.command_generator.Normalizer import Normalizer
from classes.command_generator.Generator import Generator
from make_entities import get_custom_entity_json
from helper import make_flat_entities, make_flat_templates, train_valid_test_split, generate_NumSequence


# load config values
config_file = Path('config') / 'prepare_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)

data_dir = Path(config['data_dir']) # include entity_file, template
output_dir = Path(config['output_dir'])
templates_filename = config['templates_filename']
entities_filename = config['entities_filename']
languages = config['languages']
random_seed = config['random_seed']
threshold = config['permute_thresh']

is_split = config['is_split']
test_ratio = config['split']['test_ratio']
valid_ratio = config['split']['valid_ratio']
train_ratio = config['split']['train_ratio']

is_custom = config['is_custom']
merge_type_list = config['custom']['merge_entity_types']
channel_max_range = config['custom']['channel_max_range']

num_seq_amount, num_seq_length = config['num_seq']['amount'], config['num_seq']['length']
normalizer = Normalizer().normalize_text

# save meta entities
print("Start making entities...")
time_start = time.time()
entities = pd.read_json(data_dir / entities_filename)
entities = entities[['value', 'type', 'language', 'normalizedValue', 'aliases']]
if is_custom:
    entities_merged = merge_entity_types(entities, merge_type_list)
    entities_custom = pd.DataFrame(get_custom_entity_json(languages, channel_max_range))
    entities_meta = pd.concat([entities_merged, entities_custom]).reset_index(drop=True)
else:
    entities_meta = entities.reset_index(drop=True)
entities_filename = entities_filename[:-5] + '_meta' + entities_filename[-5:]
entities_meta.to_json(data_dir / entities_filename)
time_end = time.time()

print("Augmented {} entities are saved to {}".format(len(entities_meta), data_dir / entities_filename))
print("Time used:", time_end - time_start, 's')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')


print("Reading Templates and Entities...")
df_temp = make_flat_templates(data_dir / templates_filename, languages)
df_entities = make_flat_entities(data_dir / entities_filename, languages)

print("Making Entities Special...")
df_entities_special = pd.DataFrame() 
for i in range(20):
    for lan in languages:
        num_word = Normalizer().normalize_text(str(i), lan)
        reg = '^{num_word} | {num_word}$| {num_word} '.format(num_word=num_word)
        df_entities_special = df_entities_special.append(df_entities[df_entities.value.str.contains(reg)])
df_entities_special = df_entities_special.reset_index(drop=True)
df_entities_special.to_json(data_dir / "entities_special.json")

print("Making number sequence...")
num_seq = []
for lan in languages:
    num_seq += generate_NumSequence(lan, amount=num_seq_amount, max_length=num_seq_length)
pd.DataFrame(num_seq).to_json(data_dir / "num_sequence.json", orient='records')
        
print("Making data folder: ", output_dir)
Path(output_dir).mkdir(parents=True, exist_ok=True)

if is_split:
    print("Start Templates Split...")
    df_temp_train, df_temp_valid, df_temp_test = train_valid_test_split(df_temp,
                                                                        train_ratio, valid_ratio, test_ratio,
                                                                        ['id', 'language'], ['language'])
    print("Templates Train: \n{}".format(df_temp_train['language'].value_counts()))
    print("Templates Valid: \n{}".format(df_temp_valid['language'].value_counts()))
    print("Templates Test: \n{}".format(df_temp_test['language'].value_counts()))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # make train and test for entities
    print("Start Entity Split...")
    df_entities_train, df_entities_valid, df_entities_test = train_valid_test_split(df_entities,
                                                                                    train_ratio, valid_ratio, test_ratio,
                                                                                    ['type', 'language'], ['language'])
    print("Entity Train: \n{}".format(df_entities_train['language'].value_counts()))
    print("Entity Valid: \n{}".format(df_entities_valid['language'].value_counts()))
    print("Entity Test: \n{}".format(df_entities_test['language'].value_counts()))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # Permutation
    print("Start Permutation...")
    name = 'train'
    Generator(templates=df_temp_train,
              entities=df_entities,
              name=name,
              threshold=threshold,
              normalizer=normalizer).permute(output_dir / (name + '.json'))

    
    name = 'valid'
    Generator(templates=df_temp_valid,
              entities=df_entities,
              name=name,
              threshold=threshold,
              normalizer=normalizer).permute(output_dir / (name + '.json'))
    
    name = 'test'
    Generator(templates=df_temp_test,
              entities=df_entities,
              name=name,
              threshold=threshold,
              normalizer=normalizer).permute(output_dir / (name + '.json'))
    
    name = 'train_special'
    Generator(templates=df_temp_train[df_temp_train.text.str.contains('{')],
              entities=df_entities_special,
              name=name,
              threshold=threshold,
              normalizer=normalizer).permute(output_dir / (name + '.json'))


    name = 'valid_special'
    Generator(templates=df_temp_valid[df_temp_valid.text.str.contains('{')],
              entities=df_entities_special,
              name=name,
              threshold=threshold,
              normalizer=normalizer).permute(output_dir / (name + '.json'))

    name = 'test_special'
    Generator(templates=df_temp_test[df_temp_test.text.str.contains('{')],
              entities=df_entities_special,
              name=name,
              threshold=threshold,
              normalizer=normalizer).permute(output_dir / (name + '.json'))

#     name = 'train_train'
#     Generator(templates=df_temp_train,
#               entities=df_entities_train,
#               name=name,
#               threshold=threshold,
#               normalizer=normalizer).permute(output_dir / (name + '.json'))

#     name = 'valid_valid'
#     Generator(templates=df_temp_valid,
#               entities=df_entities_valid,
#               name=name,
#               threshold=threshold,
#               normalizer=normalizer).permute(output_dir / (name + '.json'))

#     name = 'test_test'
#     Generator(templates=df_temp_test,
#               entities=df_entities_test,
#               name=name,
#               threshold=threshold,
#               normalizer=normalizer).permute(output_dir / (name + '.json'))

#     name = 'train_test'
#     Generator(templates=df_temp_train,
#               entities=df_entities_test,
#               name=name,
#               threshold=threshold,
#               normalizer=normalizer).permute(output_dir / (name + '.json'))

#     name = 'test_train'
#     Generator(templates=df_temp_test,
#               entities=df_entities_train,
#               name=name,
#               threshold=threshold,
#               normalizer=normalizer).permute(output_dir / (name + '.json'))
else:
    print("Permute tran_train...")
    name = 'train_train'
    Generator(templates=df_temp,
              entities=df_entities,
              name=name,
              threshold=threshold,
              normalizer=normalizer).permute(output_dir / (name + '.json'))