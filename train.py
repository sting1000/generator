import os

from helper import *
from tqdm import tqdm
from pathlib import Path



# augment training
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=0)
# train, _= ros.fit_resample(train, train['language'])


# take specific lan
chosen_lan = 'it'
project_path = '/mnt/workspace/project'
onmt_path = '/mnt/workspace/OpenNMT-py'
encoder_level = 'char'
decoder_level = 'char'
rnn = 'lstm' #'transformer'

train = read_data_json("data/nmt_data_json/train_train.json")
test = read_data_json("data/nmt_data_json/test_test.json")
valid = read_data_json("data/nmt_data_json/valid_valid.json")
exp_path = project_path + '/exp'
data_output_dir = exp_path + 'data/'
########


train = train[train['language'] == chosen_lan]
test = test[test['language'] == chosen_lan]
valid = valid[valid['language'] == chosen_lan]


Path(data_output_dir).mkdir(parents=True, exist_ok=True)

# generate pairs for training
for appendix in ['_char', '_token']:
    f_src_test = open(data_output_dir + 'src_test' + appendix + '.txt', "w")
    f_tgt_test = open(data_output_dir + 'tgt_test' + appendix + '.txt', "w")
    f_src_val = open(data_output_dir + 'src_val' + appendix + '.txt', "w")
    f_tgt_val = open(data_output_dir + 'tgt_val' + appendix + '.txt', "w")
    f_src_train = open(data_output_dir + 'src_train' + appendix + '.txt', "w")
    f_tgt_train = open(data_output_dir + 'tgt_train' + appendix + '.txt', "w")

    # generate
    for name, df, f_src, f_tgt in [('train', train, f_src_train, f_tgt_train),
                     ('test', test, f_src_test, f_tgt_test),
                     ('valid', valid, f_src_val, f_tgt_val)]:
        for _, row in tqdm(df.iterrows()):
            f_src.write("{}\n".format(row['src' + appendix]))
            f_tgt.write("{}\n".format(row['tgt'+ appendix]))

    f_src_val.close()
    f_tgt_val.close()
    f_src_test.close()
    f_tgt_test.close()
    f_src_train.close()
    f_tgt_train.close()

# get model name
if rnn == 'lstm':
    model_name = "BiLSTM_{encoder_level}_LSTM_{decoder_level}".format(encoder_level=encoder_level, decoder_level=decoder_level)
elif rnn == 'transformer':
    model_name = 'transformer_{encoder_level}'.format(encoder_level=encoder_level)
yaml_path = '{exp_path}/yaml/{model_name}_train.yaml'.format(exp_path=exp_path, model_name=model_name)

command_build_vocab = "python {onmt_path}/build_vocab.py -config  {yaml_path} -n_sample -1".format(onmt_path=onmt_path, yaml_path=yaml_path)
command_train = "python {onmt_path}/train.py -config {yaml_path}".format(onmt_path=onmt_path, yaml_path=yaml_path)

# train
os.system(command_build_vocab)
os.system(command_train)