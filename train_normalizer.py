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

if __name__ == "__main__":
    main()
