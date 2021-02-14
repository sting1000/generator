import argparse
import pandas as pd
import os
from utils import replace_path_in_yaml, check_folder, recover_space, prepare_onmt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from prepare.py default as ./output")
    parser.add_argument("--model_yaml", default='./models/LSTM.yaml', type=str, required=False,
                        help="Load model from OpenNMT yaml file")
    parser.add_argument("--normalizer_dir", default='./output/normalizer/LSTM', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")
    parser.add_argument("--no_classifier", default=0, type=int, required=False,
                        help="train normalizer without classifier")

    # init
    args = parser.parse_args()

    prepared_dir = args.prepared_dir
    normalizer_dir = args.normalizer_dir
    onmt_package_path = args.onmt_dir
    new_yaml_path = '{}/{}.yaml'.format(normalizer_dir, args.model_yaml.split('/')[-1])

    check_folder(normalizer_dir + '/checkpoints')
    check_folder(normalizer_dir + '/data')

    print("Preparing....")
    prepare_onmt('train', prepared_dir, normalizer_dir, args.no_classifier)
    prepare_onmt('validation', prepared_dir, normalizer_dir, args.no_classifier)
    prepare_onmt('test', prepared_dir, normalizer_dir, args.no_classifier)
    replace_path_in_yaml(yaml_path=args.model_yaml, new_yaml_path=new_yaml_path, model_path=normalizer_dir)

    command_build_vocab = "python {onmt_path}/build_vocab.py -config  {yaml_path} -n_sample -1".format(
        onmt_path=onmt_package_path, yaml_path=new_yaml_path)
    command_train = "python {onmt_path}/train.py -config {yaml_path}".format(onmt_path=onmt_package_path,
                                                                             yaml_path=new_yaml_path)

    os.system(command_build_vocab)
    os.system(command_train)

    # _, _, filenames = next(os.walk(normalizer_dir + '/checkpoints'))
    # last_model = filenames[-1]
    # print("Test on last model: ", last_model)
    #
    # model_path = normalizer_dir + '/checkpoints/{}'.format(last_model)
    # src = normalizer_dir + '/data/src_test.txt'
    # tgt = normalizer_dir + '/data/tgt_test.txt'
    # pred_path = src[:-4] + '_pred.txt'
    # command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
    #                "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path, model=model_path, src=src,
    #                                                             output=pred_path, beam_size=5)
    # print("Predicting test dataset...")
    # os.system(command_pred)
    #
    # data = pd.read_csv(pred_path, sep="\n", header=None, skip_blank_lines=False).astype(str)
    # data.columns = ["prediction_char"]
    # data['prediction_token'] = data["prediction_char"].apply(recover_space)
    # data['src'] = pd.read_csv(src, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
    # data['tgt'] = pd.read_csv(tgt, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
    # data.to_csv(normalizer_dir + '/result_test.csv', index=False)
    # correct_num = sum(data['prediction_token'] == data['tgt'])
    # print("Normalizer Error: ", len(data) - correct_num)
    # print("Normalizer Total: ", len(data))
    # print("Normalizer Accuracy: ", correct_num / len(data))


if __name__ == "__main__":
    main()
