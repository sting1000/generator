import os

project_path = '/mnt/workspace/project'
onmt_path = '/mnt/workspace/OpenNMT-py'
exp_path = project_path + '/exp'

chosen_lan = 'it'
model_name = 'BiLSTM_char_LSTM_char_it'
steps = 100000
enc_level = 'char'

output_file = 'pred_{model_name}_{steps}_steps.txt'.format(model_name=model_name, steps=steps)
src_file = '{exp_path}/data_{lan}/src_test_{enc_level}.txt'.format(lan=chosen_lan, exp_path=exp_path, enc_level=enc_level)

# train
model = '{exp_path}/{model_name}/model_step_{steps}.pt'.format(exp_path=exp_path, model_name=model_name, steps=steps)
command_pred = "python {onmt_path}/translate.py -model {model} -src {src_file} -output {output_file} -gpu 0 " \
               "-beam_size 5 -report_time".format(onmt_path=onmt_path, model=model, src_file=src_file, output_file=output_file)
os.system(command_pred)
