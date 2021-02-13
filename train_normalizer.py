import argparse
import pandas as pd
import os
from utils import replace_path_in_yaml, check_folder, recover_space, prepare_onmt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from prepare.py default as ./output")
    parser.add_argument("--model_name", default='LSTM', type=str, required=False,
                        help="Load model from OpenNMT yaml file")
    parser.add_argument("--output_dir", default='./output/normalizer', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")
    parser.add_argument("--no_classifier", default=0, type=int, required=False,
                        help="train normalizer without classifier")

    # init
    args = parser.parse_args()

    onmt_input_dir = args.prepared_dir
    onmt_output_dir = args.output_dir
    onmt_package_path = args.onmt_dir
    model_yaml_path = './models/{}.yaml'.format(args.model_name)
    new_yaml_path = '{}/{}.yaml'.format(onmt_output_dir, args.model_name)

    check_folder(onmt_output_dir + '/checkpoints')
    check_folder(onmt_output_dir + '/data')

    print("Preparing....")
    prepare_onmt('train', onmt_input_dir, onmt_output_dir, args.no_classifier)
    prepare_onmt('validation', onmt_input_dir, onmt_output_dir, args.no_classifier)
    prepare_onmt('test', onmt_input_dir, onmt_output_dir, args.no_classifier)
    replace_path_in_yaml(yaml_path=model_yaml_path, new_yaml_path=new_yaml_path, model_path=onmt_output_dir)

    command_build_vocab = "python {onmt_path}/build_vocab.py -config  {yaml_path} -n_sample -1".format(
        onmt_path=onmt_package_path, yaml_path=new_yaml_path)
    command_train = "python {onmt_path}/train.py -config {yaml_path}".format(onmt_path=onmt_package_path,
                                                                             yaml_path=new_yaml_path)

    os.system(command_build_vocab)
    os.system(command_train)

    _, _, filenames = next(os.walk(onmt_output_dir + '/checkpoints'))
    best_model = filenames[-5]
    print("Test on Best model: ", best_model)

    model = onmt_output_dir + '/checkpoints/{}'.format(best_model)
    src = onmt_output_dir + '/data/src_test.txt'
    tgt = onmt_output_dir + '/data/tgt_test.txt'
    pred_path = src[:-4] + '_pred.txt'
    command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
                   "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path, model=model, src=src,
                                                                output=pred_path, beam_size=5)
    print("Predicting test dataset...")
    os.system(command_pred)

    data = pd.read_csv(pred_path, sep="\n", header=None, skip_blank_lines=False).astype(str)
    data.columns = ["prediction_char"]
    data['prediction_token'] = data["prediction_char"].apply(recover_space)
    data['src'] = pd.read_csv(src, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
    data['tgt'] = pd.read_csv(tgt, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
    data.to_csv(onmt_output_dir + '/result_test.csv', index=False)
    correct_num = sum(data['prediction_token'] == data['tgt'])
    print("Normalizer Error: ", len(data) - correct_num)
    print("Normalizer Total: ", len(data))
    print("Normalizer Accuracy: ", correct_num / len(data))


if __name__ == "__main__":
    main()
