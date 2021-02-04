import pandas as pd
from pathlib import Path
import json
import warnings
from classes.command_generator.Generator import Generator
from classes.command_generator.Normalizer import Normalizer
from sklearn.model_selection import train_test_split
from classes.command_generator import filter_aliases
from tqdm import tqdm

warnings.filterwarnings('ignore')


def drop_small_class(df, columns, thresh=2):
    df_group = df.groupby(columns).count()
    df_group = df_group[df_group.values < thresh]
    condition = [True] * len(df)
    for targets_ind in df_group.index:
        for match in zip(columns, targets_ind):
            condition = condition & (df[match[0]] != match[1])
    return df[condition]


def permute(templates, entities, name):
    item = Generator(templates=templates,
                     entities=entities,
                     name=name,
                     threshold=threshold,
                     normalizer=normalizer)
    item.permute(output_dir / (name + '.json'))


def train_valid_test_split(df, stratify_train_remain, stratify_valid_test):
    df_train, df_test_valid = drop_and_split(df, train_ratio, stratify_train_remain)
    df_valid, df_test = drop_and_split(df_test_valid, valid_ratio / (valid_ratio + test_ratio), stratify_valid_test)
    return df_train, df_valid, df_test


def drop_and_split(df, train_size, stratify_columns):
    df = drop_small_class(df, stratify_columns)
    df_train, df_test = train_test_split(df, train_size=train_size,
                                         stratify=df[stratify_columns],
                                         random_state=random_seed)
    return df_train, df_test


def make_flat_entities(path):
    entities = pd.read_json(path)
    entities = entities[entities.language.isin(languages)]
    df_flat_entities = pd.DataFrame(columns=["value", "type", "language"])
    for _, selected_entity in tqdm(entities.iterrows()):
        aliases = selected_entity['aliases']
        value = [str(selected_entity['value'])]
        normalized_value = [selected_entity['normalizedValue']]
        lang = selected_entity['language']
        filtered_values = filter_aliases(aliases + value + normalized_value, lang)
        for val in filtered_values:
            df_flat_entities = df_flat_entities.append({
                "value": val,
                "language": lang,
                "type": selected_entity.type
            }, ignore_index=True)
    return df_flat_entities


def make_flat_templates(path):
    templates = pd.read_json(path)[['id'] + languages]
    df_flat_templates = pd.DataFrame(columns=["id", "language", "text"])
    for tem_id in tqdm(templates.id):
        for lan in languages:
            text_list = templates[templates['id'] == tem_id][lan].values[0]['texts']
            text_list = [p['ttsText'] for p in text_list]
            for text in text_list:
                df_flat_templates = df_flat_templates.append({
                    "id": tem_id,
                    "language": lan,
                    "text": text
                }, ignore_index=True)
    return df_flat_templates


# load config values
config_file = Path('config') / 'prepare_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)

data_dir = Path(config['data_dir'])
output_dir = Path(config['output_dir'])
templates_filename = config['templates_filename']
entities_filename = config['entities_filename']
languages = config['languages']
random_seed = config['random_seed']
is_split = config['is_split']
test_ratio = config['split']['test_ratio']
valid_ratio = config['split']['valid_ratio']
train_ratio = config['split']['train_ratio']
threshold = config['permute_thresh']
normalizer = Normalizer().normalize_text

print("Reading Templates and Entities...")
df_temp = make_flat_templates(data_dir / templates_filename)
df_entities = make_flat_entities(data_dir / entities_filename)

if is_split:
    print("Start Templates Split...")
    df_temp_train, df_temp_valid, df_temp_test = train_valid_test_split(df_temp, ['id', 'language'], ['language'])
    print("Templates Train: \n{}".format(df_temp_train['language'].value_counts()))
    print("Templates Valid: \n{}".format(df_temp_valid['language'].value_counts()))
    print("Templates Test: \n{}".format(df_temp_test['language'].value_counts()))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # make train and test for entities
    print("Start Entity Split...")
    df_entities_train, df_entities_valid, df_entities_test = train_valid_test_split(df_entities, ['type', 'language'],
                                                                                    ['language'])
    print("Entity Train: \n{}".format(df_entities_train['language'].value_counts()))
    print("Entity Valid: \n{}".format(df_entities_valid['language'].value_counts()))
    print("Entity Test: \n{}".format(df_entities_test['language'].value_counts()))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # Permutation
    print("Start Permutation...")
    permute(templates=df_temp_train, entities=df_entities_train, name='train_train')
    permute(templates=df_temp_valid, entities=df_entities_valid, name='valid_valid')
    permute(templates=df_temp_test, entities=df_entities_test, name='test_test')
    permute(templates=df_temp_train, entities=df_entities_test, name='train_test')
    permute(templates=df_temp_test, entities=df_entities_train, name='test_train')
else:
    print("Permute tran_train...")
    permute(templates=df_temp, entities=df_entities, name='train_train')