import pandas as pd
from pathlib import Path
import json
from generator.module import Generator
from generator.normalizer import Normalizer
import warnings

warnings.filterwarnings('ignore')

# load config values
config_file = Path('config') / 'command_generator_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)

data_dir = Path("data/nmt_data", )
output_dir = Path(config['output_dir'])
threshold = config['permute_thresh']
languages = config['languages']


def read_train_valid_test(path, prefix, appendix, lans):
    train = pd.read_csv(path / (prefix + '_train' + appendix))
    valid = pd.read_csv(path / (prefix + '_valid' + appendix))
    test = pd.read_csv(path / (prefix + '_test' + appendix))

    train = train[train['language'].isin(lans)]
    valid = valid[valid['language'].isin(lans)]
    test = test[test['language'].isin(lans)]
    return train, valid, test


# read dataframe
df_temp_train, df_temp_valid, df_temp_test = read_train_valid_test(data_dir, 'templates', '.csv', languages)
df_entities_train, df_entities_valid, df_entities_test = read_train_valid_test(data_dir, 'entities', '.csv', languages)

train_train = Generator(templates=df_temp_train, entities=df_entities_train, name='train_train', threshold=threshold)
valid_valid = Generator(templates=df_temp_valid, entities=df_entities_valid, name='valid_valid', threshold=threshold)
test_test = Generator(templates=df_temp_test, entities=df_entities_test, name='test_test', threshold=threshold)

train_train.permute(output_dir / ('train_train' + '.json'))
valid_valid.permute(output_dir / ('valid_valid' + '.json'))
test_test.permute(output_dir / ('test_test' + '.json'))

train_valid = Generator(templates=df_temp_train, entities=df_entities_valid, name='train_valid')
train_valid.tag_tuples_dic = valid_valid.tag_tuples_dic

train_test = Generator(templates=df_temp_train, entities=df_entities_test, name='train_test')
train_test.tag_tuples_dic = test_test.tag_tuples_dic

valid_train = Generator(templates=df_temp_valid, entities=df_entities_train, name='valid_train')
valid_train.tag_tuples_dic = train_train.tag_tuples_dic

test_train = Generator(templates=df_temp_test, entities=df_entities_train, name='test_train')
test_train.tag_tuples_dic = train_train.tag_tuples_dic

del valid_valid
del test_test
del train_train

train_valid.permute(output_dir / ('train_valid' + '.json'))
train_test.permute(output_dir / ('train_test' + '.json'))
valid_train.permute(output_dir / ('valid_train' + '.json'))
test_train.permute(output_dir / ('test_train' + '.json'))

