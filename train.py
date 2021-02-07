import os
from helper import *
import time
import torch
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import onmt
from onmt.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields, IterOnDevice
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.translate import GNMTGlobalScorer, Translator, TranslationBuilder
from onmt.utils.misc import set_random_seed


# load config values
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, type=str, required=True,
                        help="The configure file path e.g. config/prepare_config.json")
    parser.add_argument("--output_name", default=None, type=str, required=True,
                        help="The output filename e.g. train")


    config_file = Path('config') / 'train_config.json'
    with open(config_file, 'r', encoding='utf-8') as fp:
        configs = json.load(fp)

    for config in configs:
        model_name = config['model_name']
        chosen_lan = config['chosen_lan']
        onmt_path = config['onmt_path']
        encoder_level = config['encoder_level']
        decoder_level = config['decoder_level']
        model_type = config['model_type']
        is_split = config['is_split']
        dataset_path_list = config["dataset_path_list"]

        model_path = Path('results') / model_name
        data_output_dir = model_path / 'data'
        print("Making data folder: ", data_output_dir)
        Path(data_output_dir).mkdir(parents=True, exist_ok=True)

        yaml_path = Path('config/models') / (model_type + '.yaml')
        new_yaml_path = model_path / (model_type + '.yaml')
        replace_path_in_yaml(yaml_path, new_yaml_path, model_path)

        print("Reading data...")
        train = pd.DataFrame()
        for p in dataset_path_list:
            train = pd.concat([train, read_data_json(p)])
            print("added data from ", p)
        train = train[train['language'].isin(chosen_lan)]
        if is_split:
            test = read_data_json("data/nmt_data_json/test.json")
            valid = read_data_json("data/nmt_data_json/valid.json")
        else:
            valid = train.sample(frac=0.1, replace=True, random_state=1)
            test = train.sample(frac=0.1, replace=True, random_state=2)
        test = test[test['language'].isin(chosen_lan)]
        valid = valid[valid['language'].isin(chosen_lan)]

        train_sp = read_data_json("data/nmt_data_json/train_special.json")
        test_sp = read_data_json("data/nmt_data_json/test_special.json")
        valid_sp = read_data_json("data/nmt_data_json/valid_special.json")
        num_seq = read_data_json("data/meta_data/num_sequence.json")

        train_sp = train_sp[train_sp['language'].isin(chosen_lan)]
        test_sp = test_sp[test_sp['language'].isin(chosen_lan)]
        valid_sp = valid_sp[valid_sp['language'].isin(chosen_lan)]
        num_seq = num_seq[num_seq['language'].isin(chosen_lan)]

        make_src_tgt(train_sp, 'train_special', data_output_dir, encoder_level, decoder_level)
        make_src_tgt(train, 'train', data_output_dir, encoder_level, decoder_level)
        make_src_tgt(num_seq, 'num_seq', data_output_dir, encoder_level, decoder_level)

        make_src_tgt(valid, 'valid', data_output_dir, encoder_level, decoder_level)

        # train
        command_build_vocab = "python {onmt_path}/build_vocab.py -config  {yaml_path} -n_sample -1".format(
            onmt_path=onmt_path, yaml_path=new_yaml_path)
        command_train = "python {onmt_path}/train.py -config {yaml_path}".format(onmt_path=onmt_path,
                                                                                 yaml_path=new_yaml_path)
        os.system(command_build_vocab)
        os.system(command_train)
