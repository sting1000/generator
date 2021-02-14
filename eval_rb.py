import argparse
import pandas as pd
import requests


def rb_predict(text, language):
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = {"text": text, "language": language}
    response = requests.post('https://plato-core-postprocessor-develop.scapp-corp.swisscom.com/api/compute',
                             headers=headers, json=data)
    return eval(response.text)['text']


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from prepare.py default as ./output")
    parser.add_argument("--normalizer_dir", default='./output/normalizer/RB', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--language", default='de', type=str, required=False,
                        help="language")

    args = parser.parse_args()
    language = args.language
    normalizer_dir = args.normalizer_dir

    test = pd.read_csv('{}/test.csv'.format(args.prepared_dir),
                       converters={'token': str, 'written': str, 'spoken': str})
    data = test[['sentence_id', 'token_id', 'language', 'written', 'spoken']].drop_duplicates()
    data['tgt'] = data['written']
    data['src'] = data['spoken']
    data = data.groupby(['sentence_id']).agg({'src': ' '.join, 'tgt': ' '.join})
    data['pred'] = data['src'].apply(rb_predict, args=language)
    data.to_csv(normalizer_dir + '/normalizer_result_test.csv', index=False)

    correct_num = sum(data['pred'] == data['tgt'])
    print("Normalizer Error: ", len(data) - correct_num)
    print("Normalizer Total: ", len(data))
    print("Normalizer Accuracy: ", correct_num / len(data))


if __name__ == "__main__":
    main()
