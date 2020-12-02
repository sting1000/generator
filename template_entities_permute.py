import pandas as pd
from pathlib import Path
import json
from command_generator.Generator import Generator
from command_generator.Normalizer import Normalizer
import warnings

warnings.filterwarnings('ignore')


# def permute(templates, entities, name):
#     gen = Generator(templates=templates,
#                     entities=entities,
#                     name=name,
#                     threshold=threshold,
#                     normalizer=normalizer)
#     gen.permute(output_dir / (name + '.json'))


# load config values
config_file = Path('config') / 'command_generator_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)

data_dir = Path("data/nmt_data", )
output_dir = Path(config['output_dir'])
threshold = config['permute_thresh']
languages = config['languages']
normalizer = Normalizer().normalize_text


def read_train_valid_test(path, prefix, appendix, lans):
    train = pd.read_csv(path / (prefix + '_train' + appendix))
    valid = pd.read_csv(path / (prefix + '_valid' + appendix))
    test = pd.read_csv(path / (prefix + '_test' + appendix))

    train = train[train['language'].isin(lans)]
    valid = valid[valid['language'].isin(lans)]
    test = test[test['language'].isin(lans)]
    return train, valid, test


# read dataframe
df_temp_train, df_temp_valid, df_temp_test = read_train_valid_test(output_dir, 'templates', '.csv', languages)
df_entities_train, df_entities_valid, df_entities_test = read_train_valid_test(output_dir, 'entities', '.csv',
                                                                               languages)

# permute(templates=df_temp_train, entities=df_entities_train, name='train_train')
# permute(templates=df_temp_valid, entities=df_entities_valid, name='valid_valid')
# permute(templates=df_temp_test, entities=df_entities_test, name='test_test')
# permute(templates=df_temp_train, entities=df_entities_test, name='train_test')
# permute(templates=df_temp_test, entities=df_entities_train, name='test_train')


