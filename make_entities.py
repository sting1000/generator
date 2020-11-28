from entityMaker.manager import *
from entityMaker.helper import merge_entity_types
import pandas as pd
import json

# init
path = "./data/meta_data/"
dest_file = 'entities_custom.json'

# make entities
all_json_list = []
language_list = ['en', 'de', 'fr', 'it']
for language in language_list:
    all_json_list += generate_Duration(language)
    all_json_list += generate_RouterWiFiDuration(language)
    all_json_list += generate_TvChannelPosition(language, max_range=300)
    all_json_list += generate_RadioChannelPosition(language, max_range=300)
    all_json_list += generate_LocalsearchTimeStampEndTime(language)
    all_json_list += generate_LocalsearchTimeStampStartTime(language)
    all_json_list += generate_LocalsearchTimeStampStartDay(language)

# save self made entities
with open(path + dest_file, 'w') as fout:
    json.dump(all_json_list, fout)

# save meta entities
entities = pd.read_json(path + "entities.json")
entities = entities[['value', 'type', 'language', 'normalizedValue', 'aliases']]
entities_with_collection = merge_entity_types(entities, ["VodName", "SeriesName", "BroadcastName"])
entities_custom = pd.read_json(path + dest_file)
entities_meta = pd.concat([entities_with_collection, entities_custom]).reset_index(drop=True)
entities_meta.to_json(path + "entities_meta.json")
