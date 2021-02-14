import argparse
import os
import pandas as pd
from utils import replace_space, make_src_tgt, recover_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_dir", default='./output/pipeline_dir/distilbert_LSTM', type=str, required=False,
                        help="normalizer_dir")
    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from prepare.py default as ./output")
    parser.add_argument("--classifier_dir", default='./output/classifier/distilbert-base-german-cased', type=str,
                        required=False,
                        help="classifier_dir")
    parser.add_argument("--normalizer_dir", default='./output/normalizer/LSTM', type=str, required=False,
                        help="normalizer_dir")
    parser.add_argument("--normalizer_step", default=-1, type=int, required=False,
                        help="The steps of normalizer, default as the last one")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")
    parser.add_argument("--no_classifier", default=0, type=int, required=False,
                        help="train normalizer without classifier")

    args = parser.parse_args()
    prepared_dir = args.prepared_dir
    classifier_dir = args.classifier_dir
    normalizer_dir = args.normalizer_dir
    onmt_package_path = args.onmt_dir
    pipeline_dir = args.pipeline_dir

    # Init
    if args.normalizer_step == -1:
        _, _, filenames = next(os.walk(normalizer_dir + '/checkpoints'))
        model_name = filenames[-1]
        model = normalizer_dir + '/checkpoints/{}'.format(model_name)
    else:
        model = normalizer_dir + '/checkpoints/_step_{}.pt'.format(args.normalizer_step)
    print("Load Normalizer model at: ", model)

    if args.no_classifier:
        src_path = normalizer_dir + '/data/src_test.txt'
        tgt_path = normalizer_dir + '/data/tgt_test.txt'
        data = pd.DataFrame()
        data['src_char'] = pd.read_csv(src_path, sep="\n", header=None, skip_blank_lines=False)[0]
        data['tgt_char'] = pd.read_csv(tgt_path, sep="\n", header=None, skip_blank_lines=False)[0]
    else:
        classified_path = '{}/test_classified_pred.csv'.format(classifier_dir)
        classified_df = pd.read_csv(classified_path)
        data = classified_df[classified_df.tag != 'O'].astype(str)
        data['src_char'] = data['token'].apply(replace_space)
        data['tgt_char'] = data['src_char']

    make_src_tgt(data, 'test', data_output_dir=(pipeline_dir + '/data'), encoder_level='char',
                 decoder_level='char')
    src_path = pipeline_dir + '/data/src_test.txt'
    tgt_path = pipeline_dir + '/data/tgt_test.txt'
    pred_path = src_path[:-4] + '_pred.txt'

    print("Predicting test dataset...")
    command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
                   "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path, model=model, src=src_path,
                                                                output=pred_path, beam_size=5)
    os.system(command_pred)

    # read prediction and eval normalizer
    pred_df = pd.DataFrame()
    pred_df['pred'] = pd.read_csv(pred_path, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
    pred_df['src'] = pd.read_csv(src_path, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
    pred_df['tgt'] = pd.read_csv(tgt_path, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)

    if args.no_classifier:
        result = pred_df
    else:
        # add pred to result
        classified_df['pred'] = classified_df['token'].astype(str)
        id_TBNorm = classified_df.index[classified_df['tag'] == 'B'].tolist()
        classified_df.loc[id_TBNorm, 'pred'] = pred_df['pred'].tolist()
        result = classified_df.groupby(['sentence_id']).agg({'pred': ' '.join})

        # add label and src to result
        test_path = '{}/test.csv'.format(prepared_dir)
        test = pd.read_csv(test_path)
        test = test[['sentence_id', 'token_id', 'language', 'written', 'spoken']].drop_duplicates()
        test['src'] = test['spoken'].astype(str)
        test['label'] = test['written'].astype(str)
        test = test.groupby(['sentence_id']).agg({'src': ' '.join, 'label': ' '.join})
        result['tgt'] = test['label']
        result['src'] = test['src']

    # print result
    correct_num = sum(result['pred'] == result['tgt'])
    print("Pipeline Error: ", len(result) - correct_num)
    print("Pipeline Total: ", len(result))
    print("Pipeline Accuracy: ", correct_num / len(result))
    result.to_csv(pipeline_dir + '/pipeline_result_test.csv')


if __name__ == "__main__":
    main()
