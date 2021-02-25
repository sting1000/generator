import argparse
import pandas as pd
from tqdm import tqdm
import requests


def predict(row):
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = {"text": row['src'], "language": row['language']}
    response = requests.post('https://plato-core-postprocessor-develop.scapp-corp.swisscom.com/api/compute',
                             headers=headers, json=data)
    return eval(response.text)['text']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default='./data/museli_analysis.json', type=str, required=False,
                        help="input_path")
    parser.add_argument("--output_path", default='./data/museli_analysis_rb.json', type=str, required=False,
                        help="output_path")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    tqdm.pandas()
    df = pd.read_json(input_path)
    df['rb'] = df.progress_apply(predict, axis=1)
    df.to_json(output_path, orient="records")


if __name__ == "__main__":
    main()
