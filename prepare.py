import re
import time
from classes.entity_maker.helper import merge_entity_types
import pandas as pd
from pathlib import Path
import json
from classes.command_generator.Normalizer import Normalizer
from classes.command_generator.Generator import Generator
from helper import filter_aliases, train_test_drop_split, generate_NumSequence
import argparse
from classes.entity_maker.EntityCreator import EntityCreator
from tqdm import tqdm
from classes.command_generator.cleaning import clean_string


def get_custom_entity_json(language_list, channel_max_range):
    # custom parameters of each type of entity
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


def make_flat_templates(templates, languages):
    df_flat_templates = pd.DataFrame(columns=["id", "language", "text"])
    for tem_id in tqdm(templates.id):
        for lan in languages:
            text_list = templates[templates['id'] == tem_id][lan].values[0]['texts']
            text_list = [p['ttsText'] for p in text_list]
            for text in text_list:
                df_flat_templates = df_flat_templates.append({
                    "intent": tem_id.split('.')[1],
                    "language": lan,
                    "text": clean_string(text)
                }, ignore_index=True)
    return df_flat_templates


def make_flat_entities(entities, languages):
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
                "type": clean_string(selected_entity.type)
            }, ignore_index=True)
    return df_flat_entities


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, type=str, required=True,
                        help="The configure file path e.g. config/prepare_config.json")
    # load config values
    with open(parser.parse_args().config, 'r', encoding='utf-8') as fp:
        config = json.load(fp)

    data_dir = Path(config['data_dir'])  # include entity_file, template
    output_dir = Path(config['output_dir'])
    templates_filename = config['templates_filename']
    entities_filename = config['entities_filename']
    languages = config['languages']
    random_seed = config['random_seed']
    permute_threshold = config['permute_thresh']
    # test_ratio = config['split']['test_ratio']
    merge_type_list = config['custom']['merge_entity_types']
    channel_max_range = config['custom']['channel_max_range']
    num_seq_amount, num_seq_length = config['num_seq']['amount'], config['num_seq']['length']
    normalizer = Normalizer().normalize_text

    # reading Templates
    print("Start reading Templates...")
    df_temp = pd.read_json(data_dir / templates_filename)[['id'] + languages]
    df_temp = make_flat_templates(df_temp, languages)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # reading entities
    print("Start making entities...")
    time_start = time.time()
    entities = pd.read_json(data_dir / entities_filename)
    entities = entities[['value', 'type', 'language', 'normalizedValue', 'aliases']]
    if merge_type_list:
        entities = merge_entity_types(entities, merge_type_list)
    entities_custom = pd.DataFrame(get_custom_entity_json(languages, channel_max_range))
    entities_meta = pd.concat([entities, entities_custom]).reset_index(drop=True)
    df_entities = make_flat_entities(entities_meta, languages)
    entities_filename = entities_filename[:-5] + '_meta' + entities_filename[-5:]
    df_entities.to_json(data_dir / entities_filename, orient='records')
    time_end = time.time()
    print("Augmented {} entities are saved to {}".format(len(entities_meta), data_dir / entities_filename))
    print("Time used:", time_end - time_start, 's')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # Making Entities Special (entities with word-formatted num)
    print("Making Entities Special...")
    df_entities_special = pd.DataFrame()
    for i in range(20):  # number less than 20 cover 90%+
        for lan in languages:
            num_word = Normalizer().normalize_text(str(i), lan)
            reg = '^{num_word} | {num_word}$| {num_word} '.format(num_word=num_word)
            df_entities_special = df_entities_special.append(df_entities[df_entities.value.str.contains(reg)])
    df_entities_special = df_entities_special.reset_index(drop=True)
    df_entities_special.to_json(data_dir / "entities_special.json")

    # Making number sequence ()
    print("Making number sequence...")
    num_seq = []
    for lan in languages:
        num_seq += generate_NumSequence(lan, amount=num_seq_amount, max_length=num_seq_length)
    pd.DataFrame(num_seq).to_json(data_dir / "num_sequence.json", orient='records')

    print("Making data folder: ", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Start Permutation...")
    name = 'all_2'
    Generator(templates=df_temp,
              entities=df_entities,
              name=name,
              threshold=permute_threshold,
              normalizer=normalizer).permute(output_dir, tag=True, pad=True)


if __name__ == "__main__":
    main()
