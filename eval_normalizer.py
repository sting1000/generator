import argparse
import os
import pandas as pd
from utils import recover_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer_dir", default='./output/normalizer/LSTM', type=str, required=False,
                        help="normalizer_dir")
    parser.add_argument("--normalizer_step", default=-1, type=int, required=False,
                        help="The steps of normalizer, default as the last one")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")

    args = parser.parse_args()
    normalizer_dir = args.normalizer_dir
    onmt_package_path = args.onmt_dir

    if args.normalizer_step == -1:
        _, _, filenames = next(os.walk(normalizer_dir + '/checkpoints'))
        model_name = filenames[-1]
        model = normalizer_dir + '/checkpoints/{}'.format(model_name)
    else:
        model = normalizer_dir + '/checkpoints/_step_{}.pt'.format(args.normalizer_step)

    print("Test on last model: ", model)

    src_path = normalizer_dir + '/data/src_test.txt'
    tgt_path = normalizer_dir + '/data/tgt_test.txt'
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
    pred_df.to_csv(normalizer_dir + '/normalizer_result_test.csv', index=False)
    correct_num = sum(pred_df['pred'] == pred_df['tgt'])
    print("Normalizer Error: ", len(pred_df) - correct_num)
    print("Normalizer Total: ", len(pred_df))
    print("Normalizer Accuracy: ", correct_num / len(pred_df))