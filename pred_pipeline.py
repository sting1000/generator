import argparse
import os
import pandas as pd
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer
from datasets import Dataset
from utils import replace_space, make_src_tgt, recover_space, get_normalizer_ckpt
from train_classifier import merge_col_from_tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='example_input.txt', type=str, required=False,
                        help="input_file")
    parser.add_argument("--classifier_dir", default='./output/classifier/distilbert-base-german-cased', type=str,
                        required=False, help="classifier_dir")
    parser.add_argument("--normalizer_dir", default='./output/normalizer/LSTM', type=str, required=False,
                        help="normalizer_dir")
    parser.add_argument("--normalizer_step", default=-1, type=int, required=False,
                        help="The steps of normalizer, default as the last one")
    parser.add_argument("--onmt_dir", default='./OpenNMT-py', type=str, required=False,
                        help="OpenNMT package location")
    parser.add_argument("--no_classifier", default=0, type=int, required=False,
                        help="train normalizer without classifier")

    args = parser.parse_args()
    classifier_dir = args.classifier_dir
    normalizer_dir = args.normalizer_dir
    onmt_package_path = args.onmt_dir
    input_file = args.input_file

    # Init
    input_df = pd.read_csv(input_file, sep="\n", header=None, skip_blank_lines=False, names=['src'])
    input_df['sentence_id'] = input_df.index
    input_df['token'] = input_df['src'].str.split()
    input_df['tag'] = input_df['token'].apply(lambda x: ['O'] * len(x))
    ckpt_path = get_normalizer_ckpt(normalizer_dir, step=args.normalizer_step)
    print("Load Normalizer model at: ", ckpt_path)

    if args.no_classifier:
        data = pd.DataFrame()
        data['src_char'] = input_df['src'].apply(replace_space)
        data['tgt_char'] = input_df['src_char']
    else:
        model = AutoModelForTokenClassification.from_pretrained(classifier_dir, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(classifier_dir)
        trainer = Trainer(model=model, tokenizer=tokenizer)
        label_list = ['O', 'B-TBNorm', 'I-TBNorm']
        eval_dataset = Dataset.from_pandas(input_df)

        # tokenize
        labels = []
        tokenized_eval = tokenizer(eval_dataset["token"], truncation=True, is_split_into_words=True)
        for i, label in enumerate(eval_dataset["tag"]):
            word_ids = tokenized_eval.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                label_ids.append(label[word_idx] if word_idx != previous_word_idx else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_eval["labels"] = labels

        # predict
        predictions, labels, _ = trainer.predict(tokenized_eval)
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        df_classified = pd.DataFrame(eval_dataset)[['sentence_id', 'token']]
        df_classified['tag'] = true_predictions

        # merge tag and token
        df_classified['token'] = df_classified.apply(merge_col_from_tag, args=('token', 'tag'), axis=1)
        df_classified['tag'] = df_classified.apply(merge_col_from_tag, args=('tag', 'tag'), axis=1)
        df_classified = df_classified.set_index(['sentence_id']).apply(pd.Series.explode).reset_index()
        df_classified.to_csv(input_file[:-4] + '_classified.csv', index=False)

        # make data for normalizer
        data = df_classified[df_classified.tag != 'O'].astype(str)
        data['src_char'] = data['token'].apply(replace_space)
        data['tgt_char'] = data['src_char']

    make_src_tgt(data, 'example', data_output_dir='./tmp/data', encoder_level='char',
                 decoder_level='char')
    src_path = './tmp/data/src_example.txt'
    pred_path = src_path[:-4] + '_pred.txt'

    print("Predicting test dataset...")
    command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
                   "-beam_size {beam_size} -report_time".format(onmt_path=onmt_package_path, model=ckpt_path,
                                                                src=src_path,
                                                                output=pred_path, beam_size=5)
    os.system(command_pred)

    # read prediction and eval normalizer
    pred_df = pd.DataFrame()
    pred_df['pred'] = pd.read_csv(pred_path, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)
    pred_df['src'] = pd.read_csv(src_path, sep="\n", header=None, skip_blank_lines=False)[0].apply(recover_space)

    if args.no_classifier:
        result = pred_df
    else:
        # add pred to result
        df_classified['pred'] = df_classified['token']
        id_TBNorm = df_classified.index[df_classified['tag'] == 'B'].tolist()
        df_classified.loc[id_TBNorm, 'pred'] = pred_df['pred'].tolist()
        result = df_classified.groupby(['sentence_id']).agg({'pred': ' '.join})
        result['src'] = input_df['src']

    # print result
    result.to_csv(input_file[:-4] + '_pred.txt', index=False, header=False)


if __name__ == "__main__":
    main()
