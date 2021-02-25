import argparse
import pandas as pd
from tqdm import tqdm
from src.utils import check_folder, call_rb_API


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from dataset_prepare_generated.py default as ./output")
    parser.add_argument("--normalizer_dir", default='./output/normalizer/RB', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--language", default='de', type=str, required=False,
                        help="language")

    args = parser.parse_args()
    language = args.language
    normalizer_dir = args.normalizer_dir

    check_folder(normalizer_dir)
    test = pd.read_csv('{}/test.csv'.format(args.prepared_dir),
                       converters={'token': str, 'written': str, 'spoken': str})
    data = test[['sentence_id', 'token_id', 'language', 'written', 'spoken']].drop_duplicates()
    data['tgt'] = data['written']
    data['src'] = data['spoken']
    data = data.groupby(['sentence_id']).agg({'src': ' '.join, 'tgt': ' '.join})
    print("Start predicting...")
    tqdm.pandas()
    data['pred'] = data['src'].progress_apply(call_rb_API, args=(language,))
    data.to_csv(normalizer_dir + '/normalizer_result_test.csv', index=False)

    correct_num = sum(data['pred'] == data['tgt'])
    print("Normalizer Error: ", len(data) - correct_num)
    print("Normalizer Total: ", len(data))
    print("Normalizer Accuracy: ", correct_num / len(data))


if __name__ == "__main__":
    main()
