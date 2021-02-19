import argparse
import os
import pandas as pd
from src.utils import recover_space, get_normalizer_ckpt


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

    ckpt_path = get_normalizer_ckpt(normalizer_dir, step=args.normalizer_step)
    print("Load Normalizer model at: ", ckpt_path)

    src_path = normalizer_dir + '/data/src_test.txt'
    tgt_path = normalizer_dir + '/data/tgt_test.txt'
    pred_path = src_path[:-4] + '_pred.txt'

    print("Predicting test dataset...")
    command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} -gpu 0 " \
                   "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path,
                                                                model=ckpt_path,
                                                                src=src_path,
                                                                output=pred_path,
                                                                beam_size=5)
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


if __name__ == "__main__":
    main()
