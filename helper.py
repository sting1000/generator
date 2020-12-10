from asr_evaluation.asr_evaluation import get_error_count, get_match_count, print_diff
from edit_distance import SequenceMatcher
import pandas as pd
from command_generator.cleaning import clean_string


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


def add_pred(df, exp_path, encoder_level, decoder_level, steps, rnn):
    if rnn == 'lstm':
        model_name = "BiLSTM_{encoder_level}_LSTM_{decoder_level}".format(encoder_level=encoder_level,
                                                                          decoder_level=decoder_level)
    elif rnn == 'transformer':
        model_name = 'transformer_{encoder_level}'.format(encoder_level=encoder_level)

    path = "{exp_path}/{model_name}/pred_{steps}.txt".format(
        exp_path=exp_path,
        model_name=model_name,
        steps=steps)

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
        df.loc[index, 'entity_errors'] = sum([not clean_string(s) in hyp_line for s in row['entities_dic'].keys()])

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


def analyze(df, groupby, sort_col):
    count = df[[groupby, 'token_errors', 'token_length']].groupby(groupby).count()['token_length'].values
    meta_group = df[
        [groupby, 'char_errors', 'char_length', 'token_errors', 'token_length', 'sentence_error', 'sentence_count',
         'entity_errors', 'entity_count']].groupby(groupby).sum()
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
    df = df[df.token_errors > 0][['src_token', 'tgt_token', 'prediction']].sample(n=n, random_state=random_state)
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
    df['tgt_char'] = df.tgt_token.apply(replace_space)
    df['src_char'] = df.src_token.apply(replace_space)
    # df['entities_dic'] = df.entities_dic.apply(eval)
    return df