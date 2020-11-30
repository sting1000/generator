import pandas as pd
from pathlib import Path
import json
import warnings
from generator.module import Generator
from generator.normalizer import Normalizer
from sklearn.model_selection import train_test_split
from generator.tools import filter_aliases
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')


def drop_small_class(df, columns, thresh=2):
    df_group = df.groupby(columns).count()
    df_group = df_group[df_group.values < thresh]
    condition = [True] * len(df)
    for targets_ind in df_group.index:
        for match in zip(columns, targets_ind):
            condition = condition & (df[match[0]] != match[1])
    return df[condition]


# load config values
config_file = Path('config') / 'command_generator_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)

data_dir = Path(config['data_dir'])
output_dir = Path(config['output_dir'])
templates_filename = config['templates_filename']
entities_filename = config['entities_filename']
languages = config['languages']
id_size = config["id_size"]
extra_num_size = config['extra_num_size']
random_seed = config['random_seed']
test_ratio = config['split']['test_ratio']
valid_ratio = config['split']['valid_ratio']
train_ratio = config['split']['train_ratio']

# read dataframe
templates = pd.read_json(data_dir / templates_filename)[['id'] + languages]
entities = pd.read_json(data_dir / entities_filename)

# make train and test for templates
time_start = time.time()
gen = Generator(templates=templates, entities=entities)
df_temp = pd.DataFrame(columns=["id", "language", "text"])
for tem_id in tqdm(templates.id):
    for lan in languages:
        for text in gen.get_templates(tem_id, lan):
            df_temp = df_temp.append({
                "id": tem_id,
                "language": lan,
                "text": text
            }, ignore_index=True)
df_temp = drop_small_class(df_temp.drop_duplicates(), ['id', 'language'])
df_temp_train, df_temp_test_valid = train_test_split(df_temp,
                                                     train_size=train_ratio,
                                                     stratify=df_temp[['id', 'language']],
                                                     random_state=random_seed)
df_temp_test_valid = drop_small_class(df_temp_test_valid, ['language'])
df_temp_valid, df_temp_test = train_test_split(df_temp_test_valid,
                                               test_size=test_ratio / (valid_ratio + test_ratio),
                                               stratify=df_temp_test_valid[['language']],
                                               random_state=random_seed)
df_temp_train.to_csv(output_dir / "templates_train.csv", index=False)
df_temp_valid.to_csv(output_dir / "templates_valid.csv", index=False)
df_temp_test.to_csv(output_dir / "templates_test.csv", index=False)
time_end = time.time()
print("Templates split done:", time_end - time_start, 's')

# make train and test for entities
time_start = time.time()
entities = entities[entities.language.isin(languages)]
df_entities = pd.DataFrame(columns=["value", "type", "language"])
for _, selected_entity in tqdm(entities.iterrows()):
    aliases = selected_entity['aliases']
    value = [str(selected_entity['value'])]
    normalized_value = [selected_entity['normalizedValue']]
    lang = selected_entity['language']
    filtered_values = filter_aliases(aliases + value + normalized_value, lang)
    for v in filtered_values:
        df_entities = df_entities.append({
            "value": v,
            "language": lang,
            "type": selected_entity.type
        }, ignore_index=True)

df_entities = drop_small_class(df_entities, ['type', 'language'])
df_entities_train, df_entities_test_valid = train_test_split(df_entities,
                                                             train_size=train_ratio,
                                                             stratify=df_entities[['language']],
                                                             random_state=random_seed)
df_entities_test_valid = drop_small_class(df_entities_test_valid, ['language'])
df_entities_valid, df_entities_test = train_test_split(df_entities_test_valid,
                                                       test_size=test_ratio / (valid_ratio + test_ratio),
                                                       stratify=df_entities_test_valid[['language']],
                                                       random_state=random_seed)
# save files
df_entities_train.to_csv(output_dir / "entities_train.csv", index=False)
df_entities_valid.to_csv(output_dir / "entities_valid.csv", index=False)
df_entities_test.to_csv(output_dir / "entities_test.csv", index=False)
time_end = time.time()
print("Entities split done:", time_end - time_start, 's')
