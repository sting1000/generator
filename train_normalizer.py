import pandas as pd
import os
from utils import replace_path_in_yaml, check_folder, recover_space, prepare_onmt

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

command_build_vocab = "python {onmt_path}/build_vocab.py -config  {yaml_path} -n_sample -1".format(onmt_path=onmt_package_path, yaml_path=new_yaml_path)
command_train = "python {onmt_path}/train.py -config {yaml_path}".format(onmt_path=onmt_package_path,
                                                                         yaml_path=new_yaml_path)

os.system(command_build_vocab)
os.system(command_train)


_, _, filenames = next(os.walk(onmt_output_dir + '/checkpoints'))
best_model = filenames[-5]
print("Test on Best model: ", best_model)

model = onmt_output_dir + '/checkpoints/{}'.format(best_model)
src = onmt_output_dir + '/data/src_test.txt'
tgt = onmt_output_dir + '/data/tgt_test.txt'
pred_path = src[:-4] + '_pred.txt'
command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
               "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path, model=model, src=src,
                                                            output=pred_path, beam_size=5)
print("Predicting test dataset...")
os.system(command_pred)

data = pd.read_csv(pred_path, sep="\n", header=None, skip_blank_lines=False).astype(str)
data.columns = ["prediction_char"]
data['prediction_token'] = data["prediction_char"].apply(recover_space)
data['src'] = pd.read_csv(src, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
data['tgt'] = pd.read_csv(tgt, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
data.to_csv(onmt_output_dir + '/result_test.csv', index=False)
correct_num = sum(data['prediction_token'] == data['tgt'])
print("Normalizer Correct: ", correct_num)
print("Normalizer Total: ", len(data))
print("Normalizer Accuracy: ", correct_num/len(data))
