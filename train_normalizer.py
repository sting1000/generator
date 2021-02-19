import argparse
import os
from src.utils import make_onmt_yaml, check_folder, make_onmt_data


def main(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from dataset_prepare_generated.py default as ./output")
    parser.add_argument("--model_yaml", default='./models/LSTM.yaml', type=str, required=False,
                        help="Load model from OpenNMT yaml file")
    parser.add_argument("--normalizer_dir", default='./output/normalizer/LSTM', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")
    parser.add_argument("--no_classifier", default=0, type=int, required=False,
                        help="train normalizer without classifier")
    parser.add_argument("--encoder_level", default='char', type=str, required=False,
                        help="char or token")
    parser.add_argument("--decoder_level", default='char', type=str, required=False,
                        help="char or token")

    args = parser.parse_args(raw_args)
    prepared_dir = args.prepared_dir
    normalizer_dir = args.normalizer_dir
    onmt_package_path = args.onmt_dir
    encoder_level = args.encoder_level
    decoder_level = args.decoder_level
    no_classifier = args.no_classifier
    new_yaml_path = '{}/{}.yaml'.format(normalizer_dir, args.model_yaml.split('/')[-1])
    check_folder(normalizer_dir + '/checkpoints')
    check_folder(normalizer_dir + '/data')

    print("Preparing....")

    # make yaml and data files in normalizer_dir
    make_onmt_yaml(yaml_path=args.model_yaml, new_yaml_path=new_yaml_path, model_path=normalizer_dir)
    make_onmt_data(prepared_dir, normalizer_dir, no_classifier, encoder_level, decoder_level)

    # build covab command to use opennmt
    command_build_vocab = "python {onmt_path}/build_vocab.py " \
                          "-config  {yaml_path} " \
                          "-n_sample -1".format(onmt_path=onmt_package_path, yaml_path=new_yaml_path)

    # train command to use opennmt
    command_train = "python {onmt_path}/train.py " \
                    "-config {yaml_path}".format(onmt_path=onmt_package_path, yaml_path=new_yaml_path)

    # execute opennmt
    os.system(command_build_vocab)
    os.system(command_train)


if __name__ == "__main__":
    main()
