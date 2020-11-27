from entityMaker.maker import generate_LocalsearchTimeStampEndTime, generate_RouterWiFiDuration, \
    generate_TvChannelPosition, generate_RadioChannelPosition, generate_Duration, generate_LocalsearchTimeStampStartDay, \
    generate_LocalsearchTimeStampStartTime
from entityMaker.helper import merge_entity_types
import pandas as pd
import json

path = "./data/meta_data/"
entities = pd.read_json(path + "entities.json")
entities = entities[['value', 'type', 'language', 'normalizedValue', 'aliases']]

li = ["VodName", "SeriesName", "BroadcastName"]
entities_with_collection = merge_entity_types(entities, li)
language_list = ['en', 'de']
all_json_list = []
all_json_list += generate_Duration(language_list=language_list, entity_amount=None, amount=None)
all_json_list += generate_RouterWiFiDuration(language_list=language_list, entity_amount=None, amount=None)
all_json_list += generate_TvChannelPosition(language_list=language_list, entity_amount=None, amount=None, max_range=300)
all_json_list += generate_RadioChannelPosition(language_list=language_list, entity_amount=None, amount=None, max_range=300)
all_json_list += generate_LocalsearchTimeStampEndTime(language_list=language_list, entity_amount=None, amount=None, is_special=False)
all_json_list += generate_LocalsearchTimeStampStartTime(language_list=language_list, entity_amount=None, amount=None, is_special=False)
all_json_list += generate_LocalsearchTimeStampStartDay(language_list=language_list, entity_amount=None, amount=None)

dest_file = 'entities_custom.json'
with open(path + dest_file, 'w') as fout:
    json.dump(all_json_list, fout)
entities_custom = pd.read_json(path + dest_file)

entities_meta = pd.concat([entities_with_collection, entities_custom]).reset_index(drop=True)
entities_meta.to_json(path + "entities_meta.json")
