import os
import pandas as pd
from utils import recover_space

model_name = 'LSTM'
onmt_input_dir = './output'
onmt_output_dir = './output/normalizer'
onmt_package_path = './OpenNMT-py'
model_yaml_path = './models/{}.yaml'.format(model_name)
new_yaml_path = '{}/{}.yaml'.format(onmt_output_dir, model_name)
step = 20000

model = onmt_output_dir + '/checkpoints/_step_{step}.pt'.format(step=step)
src = onmt_output_dir + '/data/src_test.txt'
tgt = onmt_output_dir + '/data/tgt_test.txt'
pred = src[:-4] + '_pred.txt'
command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
               "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path, model=model, src=src,
                                                            output=pred, beam_size=5)
os.system(command_pred)

data = pd.read_csv(pred, sep="\n", header=None, skip_blank_lines=False).astype(str)
data.columns = ["prediction_char"]
data['prediction_token'] = data["prediction_char"].apply(recover_space)
data['src'] = pd.read_csv(src, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
data['tgt'] = pd.read_csv(tgt, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
data

classified_path = './output/test.classified_label.csv'
classified_df = pd.read_csv(classified_path)
id_tobenormalized = classified_df.index[classified_df['tag'] == 'B'].tolist()
id_remainself = classified_df.index[classified_df['tag'] == 'O'].tolist()
print('Tokens to be normalized : {}'.format(len(id_tobenormalized)))
print('Tokens to remain self : {}'.format(len(id_remainself)))

classified_df['pred'] = classified_df['token'].astype(str)
classified_df['label'] = classified_df['token'].astype(str)
classified_df.loc[id_tobenormalized, 'pred'] = data['prediction_token'].tolist()
classified_df.loc[id_tobenormalized, 'label'] = data['tgt'].tolist()
classified_df
