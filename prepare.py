import json
import argparse
import pandas as pd
from tqdm import tqdm
from src.SentenceGenerator import SentenceGenerator
from src.EntityCreator import EntityCreator
from src.utils import filter_aliases, clean_string
import random


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


def mix_entity_types(entity_df, type_list):
    if not type_list:
        return entity_df
    df_list = []
    for type_ in type_list:
        df_list.append(entity_df[entity_df.type == type_])
    collection_df = pd.concat(df_list)

    for type_ in type_list:
        collection_df['type'] = type_
        entity_df = pd.concat([entity_df, collection_df])
    entity_df = entity_df.drop_duplicates(subset=['value', 'type', 'language']).reset_index(drop=True)
    return entity_df


def prepare_templates(templates_file, languages):
    print("Preparing Templates...")
    templates = pd.read_json(templates_file)[['id'] + languages]
    df_flat_templates = pd.DataFrame(columns=["intent", "language", "text"])
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
    tqdm.pandas()
    entities = pd.read_json(entities_file)[['value', 'type', 'language', 'normalizedValue', 'aliases']]
    entities = entities[entities['language'].isin(languages)]
    entities['type'] = entities['type'].progress_apply(clean_string)

    # augment entities
    if merge_type_list:
        entities = mix_entity_types(entities, merge_type_list)
    entities_custom = pd.DataFrame(get_custom_entity_json(languages, channel_max_range))
    entities = pd.concat([entities, entities_custom]).reset_index(drop=True)

    # filter key value
    entities['value'] = entities.apply(filter_aliases, axis=1)
    entities = entities.explode('value', ignore_index=True)
    return entities[["value", "type", "language"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default='./config/prepare.json', type=str, required=False,
                        help="The configure json file path, e.g. prepare.json")
    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output will be saved to this directory default as ./output")
    parser.add_argument("--no_tagging", default=0, type=int, required=False,
                        help="disable the tag column in datasets")
    parser.add_argument("--padding", default=0, type=int, required=False,
                        help="padding size (int) to both head and tail of each sentence")
    parser.add_argument("--valid_ratio", default=0.1, type=float, required=False,
                        help="valid_ratio")
    parser.add_argument("--test_ratio", default=0.1, type=float, required=False,
                        help="test_ratio")
    parser.add_argument("--seed", default=42, type=int, required=False,
                        help="random seed")

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

    random.seed(args.seed)

    # prepare Templates and Entities
    df_templates = prepare_templates(templates_file, languages)
    df_entities = prepare_entities(entities_file, languages, merge_type_list, max_channel_range)
    gen = SentenceGenerator(templates=df_templates, entities=df_entities, max_combo_amount=max_combo_amount)
    meta_path = gen.permute(folder_path=args.prepared_dir, tagging=(1 - args.no_tagging), padding=args.padding)
    meta = pd.read_csv(meta_path, converters={'token': str, 'written': str, 'spoken': str})

    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio
    sentence_id_list = list(range(max(meta['sentence_id'])))
    random.shuffle(sentence_id_list)
    train_sep_position = int((test_ratio + valid_ratio) * len(sentence_id_list))
    test_sep_position = int(test_ratio * len(sentence_id_list))
    test_id = sentence_id_list[:test_sep_position]
    valid_id = sentence_id_list[test_sep_position:train_sep_position]
    train_id = sentence_id_list[train_sep_position:]

    test = meta[meta['sentence_id'].isin(test_id)]
    valid = meta[meta['sentence_id'].isin(valid_id)]
    train = meta[meta['sentence_id'].isin(train_id)]

    train.to_csv(args.prepared_dir + "/train.csv", index=False)
    valid.to_csv(args.prepared_dir + "/validation.csv", index=False)
    test.to_csv(args.prepared_dir + "/test.csv", index=False)


if __name__ == "__main__":
    main()
