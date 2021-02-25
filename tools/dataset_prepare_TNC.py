import pandas as pd
from tqdm import tqdm
from src.utils import check_folder
import random

valid_ratio = 0.15
test_ratio = 0.15
len_thresh = 50
prepared_dir = './TNChallenge'


def tag2bio(row):
    if row['tag'] == 'O':
        return ['O']
    else:
        # tag_list = ['B-' + row['tag']] + ['I-' + row['tag']] * (len(row['token']) - 1)
        tag_list = ['B-TBNorm'] + ['I-TBNorm'] * (len(row['token']) - 1)
        return tag_list


tqdm.pandas()
df = pd.read_csv('../TNChallenge.csv', converters={'before': str, 'after': str})
df = df.dropna()
df['after'] =df['after'].str.lower()
df['before'] =df['before'].str.lower()
df = df[~df.after.str.contains('^\W*$')]
df = df[df['class'] != 'PUNCT']
filter_id = df[df.after.apply(len) > len_thresh]['sentence_id'].unique()
df = df[~df['sentence_id'].isin(filter_id)]
df.columns = ['sentence_id', 'token_id', 'tag', 'written', 'spoken']
df["tag"].replace({"PLAIN": "O", "PUNCT": "O"}, inplace=True)
df['token'] = df['spoken'].str.split()
df['tag'] = df.apply(tag2bio, axis=1)
df = df[(df['written'] != '') & (df['spoken'] != '')]

check_folder('./TNChallenge')
meta_path = './TNChallenge/meta.csv'
with open(meta_path, 'w+') as outfile:
    columns = ['sentence_id', 'token_id', 'written', 'spoken', 'token', 'tag']
    outfile.write('\t'.join(columns) + '\n')


    def write_meta(row):
        d = dict(row)
        values = []
        for col in columns[:-2]:
            values.append(str(d[col]))

        for i in range(len(row['tag'])):
            r_values = values.copy()
            r_values.append(str(row['token'][i]))
            r_values.append(str(row['tag'][i]))
            outfile.write('\t'.join(r_values) + '\n')


    df.progress_apply(write_meta, axis=1)

meta = pd.read_csv(meta_path, sep='\t', converters={'token': str, 'written': str, 'spoken': str})
meta['language'] = 'en'

random.seed(42)
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

train.to_csv(prepared_dir + "/train.csv", index=False)
valid.to_csv(prepared_dir + "/validation.csv", index=False)
test.to_csv(prepared_dir + "/test.csv", index=False)
