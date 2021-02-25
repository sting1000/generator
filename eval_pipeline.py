import argparse
import os
import pandas as pd
from tqdm import tqdm
from src.utils import replace_space, make_src_tgt_txt, get_normalizer_ckpt, onmt_txt_to_df, call_rb_API

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_dir", default='./output/pipeline/distilbert_LSTM', type=str, required=False,
                        help="normalizer_dir")
    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from dataset_prepare_generated.py default as ./output")
    parser.add_argument("--classifier_dir", default='./output/classifier/distilbert-base-german-cased', type=str,
                        required=False, help="classifier_dir")
    parser.add_argument("--normalizer_dir", default='./output/normalizer/LSTM', type=str, required=False,
                        help="normalizer_dir")
    parser.add_argument("--encoder_level", default='char', type=str, required=False,
                        help="char or token")
    parser.add_argument("--decoder_level", default='char', type=str, required=False,
                        help="char or token")
    parser.add_argument("--normalizer_step", default=-1, type=int, required=False,
                        help="The steps of normalizer, default as the last one")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")
    parser.add_argument("--no_classifier", default=0, type=int, required=False,
                        help="train normalizer without classifier")
    parser.add_argument("--no_normalizer", default=0, type=int, required=False,
                        help="use rule-based normalizer")
    parser.add_argument("--language", default='de', type=str, required=False,
                        help="language")

    args = parser.parse_args()
    tqdm.pandas()
    prepared_dir = args.prepared_dir
    classifier_dir = args.classifier_dir
    normalizer_dir = args.normalizer_dir
    onmt_package_path = args.onmt_dir
    pipeline_dir = args.pipeline_dir
    encoder_level = args.encoder_level
    decoder_level = args.decoder_level
    no_classifier = args.no_classifier

    # Init data df
    if no_classifier:
        # read data[['src', 'tgt']]
        data = onmt_txt_to_df(normalizer_dir, encoder_level, decoder_level, have_pred=False)
    else:
        classified_path = '{}/test_classified_pred.csv'.format(classifier_dir)
        classified_df = pd.read_csv(classified_path)
        data = classified_df[classified_df['tag'] != 'O']
        data['src'], data['tgt'] = data['token'], data['src']
    data['tgt_token'], data['src_token'] = data['tgt'], data['src']
    data['tgt_char'] = data['tgt_token'].apply(replace_space)
    data['src_char'] = data['src_token'].apply(replace_space)

    # choose level data as src_test file in output dir
    make_src_tgt_txt(data, 'test', normalizer_dir=(pipeline_dir + '/data'), encoder_level=encoder_level,
                     decoder_level=decoder_level)

    if args.no_normalizer:
        print("Load Normalizer model as: Rule based...")
        data['pred'] = data['src'].progress_apply(call_rb_API, args=(args.language,))
        data['pred'] = data['pred'] if decoder_level=='token' else data['pred'].apply(replace_space)
        data[['pred']].to_csv(pipeline_dir + '/data/pred_test.txt', header=False, index=False)
    else:
        ckpt_path = get_normalizer_ckpt(normalizer_dir, step=args.normalizer_step)
        print("Load Normalizer model at: ", ckpt_path)
        print("Predicting test dataset...")
        command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} -gpu 0 " \
                       "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path,
                                                                    model=ckpt_path,
                                                                    src=pipeline_dir + '/data/src_test.txt',
                                                                    output=pipeline_dir + '/data/pred_test.txt',
                                                                    beam_size=5)
        os.system(command_pred)

    pred_df = onmt_txt_to_df(pipeline_dir, encoder_level, decoder_level)
    if no_classifier:
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
