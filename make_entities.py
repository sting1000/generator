import time
from entity_maker.manager import *
from entity_maker.helper import merge_entity_types
import pandas as pd
import json
from pathlib import Path


# custom parameters of each type of entity
def get_custom_entity_json(language_list):
    entity_list = []
    for language in language_list:
        entity_list += generate_Duration(language)
        entity_list += generate_RouterWiFiDuration(language)
        entity_list += generate_TvChannelPosition(language, max_range=300)
        entity_list += generate_RadioChannelPosition(language, max_range=300)
        entity_list += generate_LocalsearchTimeStampEndTime(language)
        entity_list += generate_LocalsearchTimeStampStartTime(language)
        entity_list += generate_LocalsearchTimeStampStartDay(language)
    return entity_list


# load config values
config_file = Path('config') / 'command_generator_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)
data_dir = Path(config['data_dir'])
templates_filename = config['templates_filename']
entities_filename = config['entities_filename']
languages = config['languages']
merge_type_list = config['merge_entity_types']

# save meta entities
print("Start making entities...")
time_start = time.time()
entities = pd.read_json(data_dir / "entities.json")
entities = entities[['value', 'type', 'language', 'normalizedValue', 'aliases']]
entities_merged = merge_entity_types(entities, merge_type_list)
entities_custom = pd.DataFrame(get_custom_entity_json(languages))
entities_meta = pd.concat([entities_merged, entities_custom]).reset_index(drop=True)
entities_meta.to_json(data_dir / entities_filename)
time_end = time.time()
print("{} entities are saved to {}".format(len(entities_meta), data_dir / entities_filename))
print("Time used:", time_end - time_start, 's')
