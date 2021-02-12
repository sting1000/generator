import pandas as pd
import os
from utils import replace_space, make_src_tgt, replace_path_in_yaml, check_folder


def prepare_onmt(name, onmt_input_dir, onmt_output_dir):
    df = pd.read_csv('{}/{}.csv'.format(onmt_input_dir, name))
    data = df[df.tag != 'O']
    data = data[['sentence_id', 'token_id', 'language', 'written', 'spoken']]
    data = data.drop_duplicates().astype(str)
    data['tgt_char'] = data.written.apply(replace_space)
    data['src_char'] = data.spoken.apply(replace_space)
    make_src_tgt(data, name, data_output_dir=(onmt_output_dir + '/data'), encoder_level='char', decoder_level='char')


model_name = 'LSTM'
onmt_input_dir = './output'
onmt_output_dir = './output/normalizer'
onmt_package_path = './OpenNMT-py'
model_yaml_path = './models/{}.yaml'.format(model_name)
new_yaml_path = '{}/{}.yaml'.format(onmt_output_dir, model_name)

check_folder(onmt_output_dir + '/checkpoints')
check_folder(onmt_output_dir + '/data')

print("Preparing....")
prepare_onmt('train', onmt_input_dir, onmt_output_dir)
prepare_onmt('validation', onmt_input_dir, onmt_output_dir)
prepare_onmt('test', onmt_input_dir, onmt_output_dir)
replace_path_in_yaml(yaml_path=model_yaml_path, new_yaml_path=new_yaml_path, model_path=onmt_output_dir)

command_build_vocab = "python {onmt_path}/build_vocab.py -config  {yaml_path} -n_sample -1".format(
    onmt_path=onmt_package_path, yaml_path=model_yaml_path)
command_train = "python {onmt_path}/train.py -config {yaml_path}".format(onmt_path=onmt_package_path,
                                                                         yaml_path=new_yaml_path)

os.system(command_build_vocab)
os.system(command_train)
