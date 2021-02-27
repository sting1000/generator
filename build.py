import argparse
import json

from src.utils import check_folder


def main():
    parser = argparse.ArgumentParser()
    # pipeline args
    parser.add_argument("--pipeline_dir", default='./output/pipeline/distilbert-base_LSTM2', type=str,
                        required=False, help="Directory to save pipeline data")
    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The prepared dataset location (containing test.csv, train.csv, validation.csv)")

    # classifier args
    parser.add_argument("--classifier_dir", default='./output/classifier/distilbert-base-uncased', type=str,
                        required=False, help="Directory to save classifier model and data")
    parser.add_argument("--pretrained", default='distilbert-base-uncased', type=str, required=False,
                        help="Load model from huggingface pretrained/ local pretrained. set None to disable classifier")

    # normalizer args
    parser.add_argument("--normalizer_dir", default='./output/normalizer/dummy', type=str, required=False,
                        help="Directory to save normalizer model and data")
    parser.add_argument("--model_yaml", default='./config/dummy.yaml', type=str, required=False,
                        help="Load normalizer model from OpenNMT yaml file")
    parser.add_argument("--encoder_level", default='char', type=str, required=False,
                        help="char or token")
    parser.add_argument("--decoder_level", default='char', type=str, required=False,
                        help="char or token")
    parser.add_argument("--language", default='en', type=str, required=False,
                        help="language of the dataset (used only for Rule-based normalizer)")
    parser.add_argument("--onmt_dir", default='../OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")

    # load config values
    args = parser.parse_args()

    # save args
    check_folder(args.pipeline_dir)
    with open(args.pipeline_dir + '/pipeline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("Pipeline args saved to: ", args.pipeline_dir + '/pipeline_args.txt')


if __name__ == "__main__":
    main()
