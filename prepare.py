import json
import argparse
import time
import pandas as pd
from tqdm import tqdm
from src.entity_helper import mix_entity_types
from src.SentenceGenerator import SentenceGenerator
from src.EntityCreator import EntityCreator
from utils import filter_aliases, clean_string


def get_custom_entity_json(language_list, channel_max_range):
    entity_list = []
    for language in language_list:
        EC = EntityCreator(language, channel_max_range)
        entity_list += EC.generate_Duration()
        entity_list += EC.generate_RouterWiFiDuration()
        entity_list += EC.generate_LocalsearchTimeStampEndTime()
        entity_list += EC.generate_LocalsearchTimeStampStartTime()
        entity_list += EC.generate_LocalsearchTimeStampStartDay()
        entity_list += EC.generate_TvChannelPosition()
        entity_list += EC.generate_RadioChannelPosition()
    return entity_list


def prepare_templates(templates_file, languages):
    print("Preparing Templates...")
    templates = pd.read_json(templates_file)[['id'] + languages]
    df_flat_templates = pd.DataFrame(columns=["id", "language", "text"])
    for tem_id in tqdm(templates.id):
        for lang in languages:
            text_list = templates[templates['id'] == tem_id][lang].values[0]['texts']
            text_list = [p['ttsText'] for p in text_list]
            for text in text_list:
                df_flat_templates = df_flat_templates.append({
                    "intent": tem_id.split('.')[1],
                    "language": lang,
                    "text": clean_string(text)
                }, ignore_index=True)
    return df_flat_templates


def prepare_entities(entities_file, languages, merge_type_list, channel_max_range):
    # init
    print("Preparing entities...")
    time_start = time.time()
    entities = pd.read_json(entities_file)[['value', 'type', 'language', 'normalizedValue', 'aliases']]
    entities = entities[entities['language'].isin(languages)]
    entities['type'] = entities['type'].apply(clean_string)

    # augment entities
    if merge_type_list:
        entities = mix_entity_types(entities, merge_type_list)
    entities_custom = pd.DataFrame(get_custom_entity_json(languages, channel_max_range))
    entities = pd.concat([entities, entities_custom]).reset_index(drop=True)

    # filter key value
    entities['value'] = entities.apply(filter_aliases, axis=1)
    entities = entities.explode('value', ignore_index=True)

    time_end = time.time()
    print("Entities Prepared: {}\t Time used: {}s".format(len(entities), time_end - time_start))
    return entities[["value", "type", "language"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, type=str, required=True,
                        help="The configure file path, e.g. config_prepare.json")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output filename e.g. data/train.csv")
    parser.add_argument("--tagging", default=0, type=int, required=False,
                        help="add tag information to output file, 1 for valid, 0 for invalid")
    parser.add_argument("--padding", default=0, type=int, required=False,
                        help="padding size (int) to head and tail of each sentence")

    # load config values
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as fp:
        config = json.load(fp)
    templates_file = config['templates_file']
    entities_file = config['entities_file']
    languages = config['languages']
    max_combo_amount = config['max_combo_amount']
    merge_type_list = config['mix_entity_types']
    max_channel_range = config['max_channel_range']

    # prepare Templates and Entities
    df_templates = prepare_templates(templates_file, languages)
    df_entities = prepare_entities(entities_file, languages, merge_type_list, max_channel_range)

    gen = SentenceGenerator(templates=df_templates, entities=df_entities, max_combo_amount=max_combo_amount)
    gen.permute(output_file=args.output_file, tagging=args.tagging, padding=args.padding)


if __name__ == "__main__":
    main()
