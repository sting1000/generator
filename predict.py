import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import read_data_json, get_cer, get_ser, get_wer, add_pred, print_errors

def get_eval_item(df, step, mode):
    p = {
         "mode": mode,
         "size":len(df),
         "step": step,
         "wer":  get_wer(df),
         "ser": get_ser(df), 
         "eer": 0, #get_eer(df), 
         "cer": get_cer(df)}
    return p

# set path
config_file = Path('config') / 'predict_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    configs = json.load(fp)
    
for config in configs:
    input_path = config['input_path']
    output_path = config['output_path']
    onmt_path = config['onmt_path']
    chosen_lan = config['chosen_lan']
    model_name = config['model_name']
    steps = config['steps']
    enc_level = config['encoder_level']
    dec_level = config['decoder_level']
    beam_size = config['beam_size']

    # generate
    test = read_data_json(input_path)
    test_new = test[test['language'].isin(chosen_lan)]
    f_src_test = open('src_test.txt', "w")
    f_tgt_test = open('tgt_test.txt', "w")
    for _, row in tqdm(test_new.iterrows()):
        f_src_test.write("{}\n".format(row['src_' + enc_level]))
    f_src_test.close()

    # init
    result = pd.DataFrame(columns=["mode", "size", "step", "wer", "ser", "eer", "cer"])
    data = read_data_json(input_path)
    data = data[data['language'].isin(chosen_lan)]
    data.reset_index(drop=True, inplace=True)

    # pred 0 step
    # char2char = add_pred(data, None, dec_level)
    # result = result.append(get_eval_item(char2char, 0, mode='overall'), ignore_index=True)
    # result = result.append(get_eval_item(char2char[char2char.src_token != char2char.tgt_token], 0, mode='diff'), ignore_index=True)
    # result = result.append(get_eval_item(char2char[char2char.src_token == char2char.tgt_token], 0, mode='equal'), ignore_index=True)

    # predict given list of steps
    for st in tqdm(steps):
        model = Path('results') / model_name / '_step_{steps}.pt'.format(steps=st)
        command_pred = "python {onmt_path}/translate.py -model_name {model} -src src_test.txt -output {output_path} -gpu 0 " \
                       "-beam_size {beam_size} -report_time".format(onmt_path=onmt_path, model=model,
                                                                    output_path=output_path, beam_size=beam_size)
        os.system(command_pred)

        char2char = add_pred(data, output_path, dec_level)
        result = result.append(get_eval_item(char2char, st, mode='overall'), ignore_index=True)
        result = result.append(get_eval_item(char2char[char2char.src_token != char2char.tgt_token], st, mode='diff'), ignore_index=True)
        result = result.append(get_eval_item(char2char[char2char.src_token == char2char.tgt_token], st, mode='equal'), ignore_index=True)

    #result.to_csv('results/result_{}.csv'.format('_'.join(chosen_lan)), index=False)

    print_errors(char2char[char2char.src_token != char2char.tgt_token], n=1, random_state=3)

    #     print("----Overall Plain-----")
    #     char2char = add_pred(data, None, dec_level)
    #     print("WER: {}%".format(get_wer(char2char)))
    #     print("SER: {}%".format(get_ser(char2char)))
    #     print("EER: {}%".format(get_eer(char2char)))
    #     print("CER: {}%\n".format(get_cer(char2char)))

    #     print("----Special Plain----")
    #     print("WER: {}%".format(get_wer(char2char[char2char.src_token != char2char.tgt_token])))
    #     print("SER: {}%".format(get_ser(char2char[char2char.src_token != char2char.tgt_token])))
    #     print("EER: {}%".format(get_eer(char2char[char2char.src_token != char2char.tgt_token])))
    #     print("CER: {}%\n".format(get_cer(char2char[char2char.src_token != char2char.tgt_token])))

    #     print("----Copy Plain----")
    #     print("WER: {}%".format(get_wer(char2char[char2char.src_token == char2char.tgt_token])))
    #     print("SER: {}%".format(get_ser(char2char[char2char.src_token == char2char.tgt_token])))
    #     print("EER: {}%".format(get_eer(char2char[char2char.src_token == char2char.tgt_token])))
    #     print("CER: {}%\n".format(get_cer(char2char[char2char.src_token == char2char.tgt_token])))
    #     print("----Overall Model-----")
    #     char2char = add_pred(data, output_path, dec_level)
    #     print("WER: {}%".format(get_wer(char2char)))
    #     print("SER: {}%".format(get_ser(char2char)))
    #     print("EER: {}%".format(get_eer(char2char)))
    #     print("CER: {}%\n".format(get_cer(char2char)))
    #     print_errors(char2char, n=5, random_state=3)

    #     print("----Special Model-----")
    #     print("WER: {}%".format(get_wer(char2char[char2char.src_token != char2char.tgt_token])))
    #     print("SER: {}%".format(get_ser(char2char[char2char.src_token != char2char.tgt_token])))
    #     print("EER: {}%".format(get_eer(char2char[char2char.src_token != char2char.tgt_token])))
    #     print("CER: {}%\n".format(get_cer(char2char[char2char.src_token != char2char.tgt_token])))
    #     print_errors(char2char[char2char.src_token != char2char.tgt_token], n=5, random_state=3)

    #     print("----Copy Model----")
    #     print("WER: {}%".format(get_wer(char2char[char2char.src_token == char2char.tgt_token])))
    #     print("SER: {}%".format(get_ser(char2char[char2char.src_token == char2char.tgt_token])))
    #     print("EER: {}%".format(get_eer(char2char[char2char.src_token == char2char.tgt_token])))
    #     print("CER: {}%\n".format(get_cer(char2char[char2char.src_token == char2char.tgt_token])))
    #     print_errors(char2char[char2char.src_token == char2char.tgt_token], n=5, random_state=3)