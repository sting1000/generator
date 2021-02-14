import pandas as pd
from tqdm import tqdm
import requests


def predict(row):
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = {"text": row['src'], "language": row['language']}
    response = requests.post('https://plato-core-postprocessor-develop.scapp-corp.swisscom.com/api/compute',
                             headers=headers, json=data)
    return eval(response.text)['text']


data_path = './data/museli_analysis.json'
output_path = './data/museli_analysis_rb.json'
df = pd.read_json(data_path)

tqdm.pandas()
df['rb'] = df.progress_apply(predict, axis=1)
df.to_json(output_path, orient="records")
