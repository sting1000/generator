import argparse
import json
import random
import time

import pandas as pd
import requests
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='./data/museli_analysis_rb.json', type=str, required=False,
                        help="input_path")
    parser.add_argument("--output_file", default='./example_result.csv', type=str, required=False,
                        help="input_path")
    parser.add_argument("--mode", default='make', type=str, required=False,
                        help="make/ pred")
    parser.add_argument("--pred_file", default='./example_intput.txt', type=str,
                        required=False,
                        help="The output dir from make_Generated_dataset.py default as ./output")
    parser.add_argument("--pipeline_dir", default='./output/pipeline/distilbert_LSTM', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--language", default='de', type=str, required=False,
                        help="language")

    args = parser.parse_args()
    pred_file = args.pred_file
    output_file = args.output_file
    language = args.language
    pipeline_dir = args.pipeline_dir
    input_file = args.input_file
    mode = args.mode

    tqdm.pandas()
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

    input_df = pd.read_json(input_file)
    input_df = input_df[input_df['language'] == language]

    if mode == 'make':
        input_df[['src']].to_csv(pred_file, header=False, index=False)

    elif mode == 'pred':
        # read pred
        pred = pd.read_csv(pred_file, names=['ai'], sep="\n", header=None, skip_blank_lines=False)
        input_df['ai'] = pred['ai']

        for model in ['rb', 'ai']:
            intent_col = 'intent_' + model
            entity_col = 'entities_type_' + model
            nlu_res = pd.DataFrame(input_df[model].progress_apply(predict_nlu, args=(language,)).to_list(),
                                   columns=[intent_col, entity_col])
            input_df = pd.concat([input_df, nlu_res], axis=1)
            input_df[intent_col].replace(to_replace=oter_replacement, inplace=True)
            input_df[entity_col].replace(to_replace=oter_replacement, inplace=True)

            condition_intent = input_df['intent'] == input_df[intent_col]
            condition_entity = input_df['entities_type'] == input_df[entity_col]
            ote = len(input_df) - len(input_df[condition_entity][condition_intent])
            oter = ote / len(input_df)

            print(model + " Model:")
            print("Size: ", len(input_df))
            print("OTE: ", ote)
            print("OTER: ", oter)
        input_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
