import argparse, os
from src.utils import get_normalizer_ckpt, read_onmt_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer_dir", default='./output/normalizer/LSTM', type=str, required=False,
                        help="normalizer_dir")
    parser.add_argument("--normalizer_step", default=-1, type=int, required=False,
                        help="The steps of normalizer, default as the last one")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")
    parser.add_argument("--encoder_level", default='char', type=str, required=False,
                        help="char or token")
    parser.add_argument("--decoder_level", default='char', type=str, required=False,
                        help="char or token")

    args = parser.parse_args()
    normalizer_dir = args.normalizer_dir
    onmt_package_path = args.onmt_dir
    encoder_level = args.encoder_level
    decoder_level = args.decoder_level
    ckpt_path = get_normalizer_ckpt(normalizer_dir, step=args.normalizer_step)
    print("Loaded Normalizer model at: ", ckpt_path)

    print("Predicting test dataset...")
    src_path = normalizer_dir + '/data/src_test.txt'
    pred_path = normalizer_dir + '/data/pred_test.txt'
    command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} -gpu 0 " \
                   "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path,
                                                                model=ckpt_path,
                                                                src=src_path,
                                                                output=pred_path,
                                                                beam_size=5)
    os.system(command_pred)

    # read prediction and eval normalizer
    pred_df = read_onmt_text(normalizer_dir, encoder_level, decoder_level)
    result_path = normalizer_dir + '/normalizer_result_test.csv'
    print("Results save to: ", result_path)
    pred_df.to_csv(result_path, index=False)

    correct_num = sum(pred_df['pred'] == pred_df['tgt'])
    print("Normalizer Error: ", len(pred_df) - correct_num)
    print("Normalizer Total: ", len(pred_df))
    print("Normalizer Accuracy: ", correct_num / len(pred_df))


if __name__ == "__main__":
    main()
