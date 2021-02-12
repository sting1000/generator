import os
import pandas as pd
from utils import  replace_space, make_src_tgt, recover_space

onmt_input_dir = './output'
onmt_output_dir = './output/normalizer'
onmt_package_path = './OpenNMT-py'
name = 'test'

pipeline_output_dir = './output/pipeline'

# Init
classified_path = '{}/{}_classified_pred.csv'.format(onmt_input_dir, name)
_, _, filenames = next(os.walk(onmt_output_dir + '/checkpoints'))
best_model = filenames[-5]
model = onmt_output_dir + '/checkpoints/{}'.format(best_model)
print("Load Best model: ", best_model)

classified_df = pd.read_csv(classified_path)
data = classified_df[classified_df.tag != 'O'].astype(str)
data['src_char'] = data['token'].apply(replace_space)
data['tgt_char'] = data['src_char']
make_src_tgt(data, 'test', data_output_dir=(pipeline_output_dir + '/data'), encoder_level='char', decoder_level='char')

src = pipeline_output_dir + '/data/src_test.txt'
pred_path = src[:-4] + '_pred.txt'

print("Predicting test dataset...")
command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
               "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path, model=model, src=src,
                                                            output=pred_path, beam_size=5)
os.system(command_pred)

# read prediction to pred_df
pred_df = pd.read_csv(pred_path, sep="\n", header=None, skip_blank_lines=False).astype(str)
pred_df.columns = ["prediction_char"]
pred_df['prediction_token'] = pred_df["prediction_char"].apply(recover_space)

# add pred to result
classified_df['pred'] = classified_df['token'].astype(str)
id_TBNorm = classified_df.index[classified_df['tag'] == 'B'].tolist()
classified_df.loc[id_TBNorm, 'pred'] = pred_df['prediction_token'].tolist()
result = classified_df.groupby(['sentence_id']).agg({'pred': ' '.join})

# add label and src to result
test_path = '{}/{}.csv'.format(onmt_input_dir, name)
test = pd.read_csv(test_path)
test = test[['sentence_id', 'token_id', 'language', 'written', 'spoken']].drop_duplicates()
test['src'] = test['spoken'].astype(str)
test['label'] = test['written'].astype(str)
test = test.groupby(['sentence_id']).agg({'src': ' '.join, 'label': ' '.join})
result['label'] = test['label']
result['src'] = test['src']

# print result
correct_num = sum(result['pred'] != result['label'])
print("Pipeline Correct: ", correct_num)
print("Pipeline Total: ", len(result))
print("Pipeline Accuracy: ", correct_num/len(result))
result.to_csv(pipeline_output_dir + '/pipeline_result.csv')