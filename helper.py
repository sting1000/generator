from asr_evaluation.asr_evaluation import get_error_count, get_match_count, print_diff
from edit_distance import SequenceMatcher
import pandas as pd
from classes.command_generator.cleaning import clean_string
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import random
from classes.command_generator.Normalizer import Normalizer


def replace_space(s: str):
    s = s.replace(' ', '_')
    return ' '.join(list(s))


def recover_space(s: str):
    s = s.replace(' ', '')
    s = s.replace('_', ' ')
    return s


def read_data(path):
    df = pd.read_csv(path).drop_duplicates()
    df.columns = ['id', 'language', 'src_token', 'tgt_token', 'entities_dic']
    df['src_token'] = df['src_token'].str.lower()
    df['tgt_token'] = df['tgt_token'].str.lower()
    df['tgt_char'] = df.tgt_token.apply(replace_space)
    df['src_char'] = df.src_token.apply(replace_space)
    df['entities_dic'] = df.entities_dic.apply(eval)
    return df


def make_command(exp_path, encoder_level, decoder_level, steps, rnn):
    if rnn == 'lstm':
        model_name = "BiLSTM_{encoder_level}_LSTM_{decoder_level}".format(encoder_level=encoder_level,
                                                                          decoder_level=decoder_level)
    elif rnn == 'transformer':
        model_name = 'transformer_{encoder_level}'.format(encoder_level=encoder_level)

    with open("translate.sh", 'w') as f:
        command = "onmt_build_vocab --config {exp_path}/yaml/{model_name}_prep.yaml -n_sample -1".format(
            exp_path=exp_path, model_name=model_name)
        print(command)
        f.write(command)
        print()

        command = "onmt_train --config {exp_path}/yaml/{model_name}_train.yaml".format(exp_path=exp_path,
                                                                                       model_name=model_name)
        print(command)
        f.write(command)
        print()

        command = "onmt_translate -model {exp_path}/{model_name}/model_step_{steps}.pt -src {exp_path}/data/src_test_{encoder_level}.txt -output {exp_path}/{model_name}/pred_{steps}.txt -gpu 0 -beam_size 5 -report_time".format(
            exp_path=exp_path, encoder_level=encoder_level, model_name=model_name, steps=steps)
        print(command)
        f.write(command)
        print()
    f.close()


def add_pred(df, path, decoder_level):
    if not path:
        data = df[['src_char']].reset_index(drop=True)
        data = data.fillna('')
    else:
        data = pd.read_csv(path, sep="\n", header=None, skip_blank_lines=False)
        data = data.fillna('')
    data.columns = ["prediction"]

    df = df.reset_index(drop=True)
    if decoder_level == 'char':
        df['prediction_char'] = data["prediction"]
        df["prediction"] = data["prediction"].apply(recover_space)
    else:
        df['prediction_char'] = data["prediction"].apply(replace_space)
        df["prediction"] = data["prediction"]

    errors, matches, ref_length = [], [], []
    errors_char, matches_char, ref_length_char = [], [], []
    df['entity_errors'] = 0
    for index, row in df.iterrows():
        # token
        ref_line = row['tgt_token']
        hyp_line = row['prediction']
        ref = ref_line.split()
        hyp = hyp_line.split()
        sm = SequenceMatcher(a=ref, b=hyp)
        errors.append(get_error_count(sm))
        matches.append(get_match_count(sm))
        ref_length.append(len(ref))

        # char
        ref = row['tgt_char'].split()
        hyp = row['prediction_char'].split()
        sm = SequenceMatcher(a=ref, b=hyp)
        errors_char.append(get_error_count(sm))
        matches_char.append(get_match_count(sm))
        ref_length_char.append(len(ref))

        # entity
        df.loc[index, 'entity_errors'] = 0  # sum([not clean_string(s) in hyp_line for s in row['entities_dic'].keys()])

    df['entity_count'] = df['entities_dic'].apply(len)

    df['token_errors'] = errors
    df['token_matches'] = matches
    df['token_length'] = ref_length

    df['char_errors'] = errors_char
    df['char_matches'] = matches_char
    df['char_length'] = ref_length_char

    df['sentence_count'] = 1
    df['sentence_error'] = 0
    df.loc[df['token_errors'] > 0, 'sentence_error'] = 1
    return df


def analyze(df, group_by, sort_col):
    count = df[[group_by, 'token_errors', 'token_length']].groupby(group_by).count()['token_length'].values
    meta_group = df[
        [group_by, 'char_errors', 'char_length', 'token_errors', 'token_length', 'sentence_error', 'sentence_count',
         'entity_errors', 'entity_count']].groupby(group_by).sum()
    meta_group['wer'] = round(100 * meta_group.token_errors / meta_group.token_length, 2)
    meta_group['ser'] = round(100 * meta_group.sentence_error / meta_group.sentence_count, 2)
    meta_group['eer'] = round(100 * meta_group.entity_errors / meta_group.entity_count, 2)
    meta_group['cer'] = round(100 * meta_group.char_errors / meta_group.char_length, 2)
    meta_group = meta_group.reset_index().sort_values(sort_col, ascending=False)
    return meta_group.reset_index(drop=True)


def get_wer(df):
    return round(100 * sum(df.token_errors) / sum(df.token_length), 2)


def get_ser(df):
    return round(100 * sum(df.sentence_error) / sum(df.sentence_count), 2)


def get_eer(df):
    return round(100 * sum(df.entity_errors) / sum(df.entity_count), 2)


def get_cer(df):
    return round(100 * sum(df.char_errors) / sum(df.char_length), 2)


def print_errors(df, n, random_state=1):
    df = df[df.token_errors > 0][['src_token', 'tgt_token', 'prediction']]
    print(len(df))
    df = df.sample(frac=n, random_state=random_state)
    for src_line, ref_line, hyp_line in zip(df['src_token'].values, df['tgt_token'].values, df['prediction'].values):
        ref = ref_line.split()
        hyp = hyp_line.split()
        sm = SequenceMatcher(a=ref, b=hyp)
        print("SRC:", src_line)
        print_diff(sm, ref, hyp)
        print()


def read_data_json(path):
    df = pd.read_json(path)  # .drop_duplicates()
    df.columns = ['id', 'language', 'src_token', 'tgt_token', 'entities_dic']
    df['src_token'] = df['src_token'].astype(str).str.lower()
    df['tgt_token'] = df['tgt_token'].astype(str).str.lower()
    df['tgt_char'] = df.tgt_token.apply(replace_space)
    df['src_char'] = df.src_token.apply(replace_space)
    # df['entities_dic'] = df.entities_dic.apply(eval)
    return df


def dump_json(outfile, data, has_no_output):
    if data:
        if has_no_output:
            outfile.write(json.dumps(data))
        else:
            outfile.write(",")
            outfile.write(json.dumps(data))
        has_no_output = False
    return has_no_output


# generate pairs for training
def make_src_tgt(df, df_type, data_output_dir, encoder_level, decoder_level):
    print("Making src tgt for: ", df_type)
    f_src = open(data_output_dir / ('src_' + df_type + '.txt'), "w")
    f_tgt = open(data_output_dir / ('tgt_' + df_type + '.txt'), "w")
    for _, row in tqdm(df.iterrows()):
        f_src.write("{}\n".format(row['src_' + encoder_level]))
        f_tgt.write("{}\n".format(row['tgt_' + decoder_level]))
    f_src.close()
    f_tgt.close()


def replace_path_in_yaml(yaml_path, new_yaml_path, model_path):
    with open(yaml_path) as f:
        lines = f.readlines()
    with open(new_yaml_path, "w+") as f:
        for ind, l in enumerate(lines):
            lines[ind] = re.sub("\{\*path\}", str(model_path), l)
        f.writelines(lines)


def drop_small_class(df, columns, thresh=2):
    df_group = df.groupby(columns).count()
    df_group = df_group[df_group.values < thresh]
    condition = [True] * len(df)
    for targets_ind in df_group.index:
        for match in zip(columns, targets_ind):
            condition = condition & (df[match[0]] != match[1])
    return df[condition]


def train_test_drop_split(df, test_size, stratify_columns, random_state=42):
    df = drop_small_class(df, stratify_columns)
    df_train, df_test = train_test_split(df, test=test_size, stratify=df[stratify_columns], random_state=random_state)
    return df_train, df_test


def filter_aliases(row) -> list:
    """
    Filters list of aliases to keep only the useful ones.
    It is used to remove all the noisy aliases given by tv that are useful for ASR.

    E.g. ['s r f 1', 'SRF 1', 'srf eins'] becomes ['srf 1']
    """
    aliases = row['aliases'] + [str(row['value'])]
    language = row['language']

    regex = re.compile(r'\b[a-zA-Z]\b')
    for item in aliases:
        item = str(item)
        if regex.findall(item):  # modified
            upper_alias = restore_abbreviations_in_text(text=item, uppercase=True).strip()
            aliases.remove(item)
            if upper_alias not in aliases:
                aliases.append(upper_alias)

    # remove norm duplication
    aliases_set = set([clean_string(x) for x in aliases])
    normalized_aliases_set = set()
    for item in aliases_set:
        norm = Normalizer().normalize_text(item, language)
        if norm != item:
            normalized_aliases_set.add(norm)
    filtered_values = list(aliases_set - normalized_aliases_set)
    return filtered_values


def restore_abbreviations_in_text(text: str, uppercase=False) -> str:
    """
    Restores malformed abbreviations in text.
    E.g. 'Go to S R F 1' becomes 'Go to SRF 1'.
    """
    abbreviations = find_space_separated_abbreviations(text=text)
    if abbreviations:
        for abbreviation in abbreviations:
            if uppercase:
                text = text.replace(' '.join(list(abbreviation)), abbreviation.upper())
            else:
                text = text.replace(' '.join(list(abbreviation)), abbreviation)
    return text


def find_space_separated_abbreviations(text: str) -> list:
    """
    Finds abbreviations in text written with space among their letters.
    E.g. 'Go to S R F 1' finds 'SRF' as abbreviation.
    """
    regex = re.compile(r'\b[a-zA-Z]\b')

    # initialize values
    abbreviations = []
    abbreviation = ''
    last_pos = -1

    for item in regex.finditer(text):
        if last_pos == -1:
            abbreviation += item.group()
            last_pos = item.span()[1]
        elif item.span()[0] == last_pos + 1:
            abbreviation += item.group()
            last_pos = item.span()[1]
        elif len(abbreviation) > 1:
            abbreviations.append(abbreviation)
            abbreviation = item.group()
            last_pos = -1
        elif len(abbreviation) == 1:
            abbreviation = item.group()
            last_pos = -1

    # append last found abbreviation
    if len(abbreviation) > 1:
        abbreviations.append(abbreviation)

    return abbreviations


def generate_NumSequence(language, amount=3000, max_length=12):
    entity_list = []
    entity_type = "NumberSequence"
    for i in tqdm(range(amount)):
        length = random.randint(3, max_length)
        low = 10 ** length
        high = low * 10 - 1
        value = str(random.randint(low, high))
        item = {
            "type": entity_type,
            "language": language,
            "spoken": Normalizer().normalize_text(' '.join(list(value)), language),
            "written": value,
            "entities_dic": []
        }
        entity_list.append(item)
    return entity_list
