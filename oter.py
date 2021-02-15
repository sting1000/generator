import argparse
import json
import random
import time

import pandas as pd
import requests
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_file", default='./example_output.txt', type=str,
                        required=False,
                        help="The output dir from dataset_prepare.py default as ./output")
    parser.add_argument("--pipeline_dir", default='./output/pipeline/distilbert_LSTM', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--language", default='de', type=str, required=False,
                        help="language")

    args = parser.parse_args()
    tqdm.pandas()
    pred_file = args.pred_file
    language = args.language
    pipeline_dir = args.pipeline_dir
    museli_json = './data/museli_analysis_rb.json'
    oter_replacement = {'SeriesName': 'Video',
                        'VodName': 'Video',
                        'BroadcastName': 'Video',
                        'DirectTvSearch': 'TvSearch'}

    def check_resp(resp):
        cond = ('intents' in resp) and ('entities' in resp)
        return cond

    def predict_nlu(text, language):
        id = str(random.random())[2:]
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'tenant': 'tv'}
        params = (('callType', 'DIALOG_ENGINE'),)
        data = {"contextId": "test-{}".format(id),
                "traceId": "trace-{}".format(id),
                "text": text,
                "language": language,
                "contextData": []}
        response = requests.post('https://plato-api-nlu-develop.scapp-corp.swisscom.com/solutions/tv/compute',
                                 headers=headers, params=params, json=data)

        patience = 5
        try:
            val = json.loads(response.text)
        except:
            patience = 0
            print(response)

        while patience and not check_resp(val):
            time.sleep(1)
            response = requests.post('https://plato-api-nlu-develop.scapp-corp.swisscom.com/solutions/tv/compute',
                                     headers=headers, params=params, json=data)
            val = json.loads(response.text)
            patience -= 1

        if patience:
            intent = val['intents'][0]['value'] if val['intents'] else 'Unknown'
            entities_type = val['entities'][0]['type'] if val['entities'] else 'Unknown'
        else:
            intent = 'NoResponse'
            entities_type = 'NoResponse'
        return [intent, entities_type]

    museli = pd.read_json(museli_json)
    museli = museli[museli['language'] == language]
    # museli[['src']].to_csv('example_input.txt', header=False, index=False)

    # read pred
    pred = pd.read_csv(pred_file, names=['ai'], sep="\n", header=None, skip_blank_lines=False)
    museli['ai'] = pred['ai']

    for mode in ['rb', 'ai']:
        intent_col = 'intent_' + mode
        entity_col = 'entities_type_' + mode
        nlu_res = pd.DataFrame(museli[mode].progress_apply(predict_nlu, args=(language,)).to_list(),
                               columns=[intent_col, entity_col])
        museli = pd.concat([museli, nlu_res], axis=1)
        museli[intent_col].replace(to_replace=oter_replacement, inplace=True)
        museli[entity_col].replace(to_replace=oter_replacement, inplace=True)

        condition_intent = museli['intent'] == museli[intent_col]
        condition_entity = museli['entities_type'] == museli[entity_col]
        ote = len(museli) - len(museli[condition_entity][condition_intent])
        oter = ote / len(museli)

        print(mode + " Model:")
        print("Size: ", len(museli))
        print("OTE: ", ote)
        print("OTER: ", oter)
    museli.to_csv(pipeline_dir + '/result_museli_oter_test.csv', index=False)


if __name__ == "__main__":
    main()
