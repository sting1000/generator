import argparse
import os
import pandas as pd
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer
from datasets import Dataset, Sequence, ClassLabel, DatasetDict
from src.utils import replace_space, make_onmt_txt, recover_space, get_normalizer_ckpt, clean_string
from train_classifier import merge_col_from_tag
from transformers import DataCollatorForTokenClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='./example/example_input.txt', type=str, required=False,
                        help="input_file")
    parser.add_argument("--output_file", default='./example/example_output.txt', type=str, required=False,
                        help="output_file")
    parser.add_argument("--classifier_dir", default='./output/classifier/distilbert-base-german-cased', type=str,
                        required=False, help="classifier_dir")
    parser.add_argument("--tokenizer_dir", default='distilbert-base-german-cased', type=str,
                        required=False, help="tokenizer_dir")
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
    output_file = args.output_file

    # Init
    input_df = pd.read_csv(input_file, sep="\n", header=None, skip_blank_lines=False, names=['src'])
    input_df['sentence_id'] = input_df.index
    input_df['src'] = input_df['src'].apply(clean_string)
    input_df['token'] = input_df['src'].str.split()
    input_df['tag'] = input_df['token'].apply(lambda x: ['O'] * len(x))

    if args.no_classifier:
        data = pd.DataFrame()
        data['src_char'] = input_df['src'].apply(replace_space)
        data['tgt_char'] = input_df['src_char']
    else:
        model = AutoModelForTokenClassification.from_pretrained(classifier_dir, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
        label_list = ['O', 'B-TBNorm', 'I-TBNorm']
        feature_tag = Sequence(ClassLabel(num_classes=3, names=label_list))
        input_df['tag'] = input_df['tag'].apply(feature_tag.feature.str2int)
        eval_dataset = Dataset.from_pandas(input_df)
        eval_dataset.features["tag"] = feature_tag
        datasets = DatasetDict({
            'eval': eval_dataset
        })

        def tokenize_and_align_labels(examples):
            labels = []
            label_all_tokens = False
            tokenized_inputs = tokenizer(examples["token"], truncation=True, is_split_into_words=True)

            for i, label in enumerate(examples[f"tag"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

        # predict
        predictions, labels, _ = trainer.predict(tokenized_datasets['eval'])
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

    # TODO: change encoder decoder level
    make_onmt_txt(data, 'example', data_output_dir='./tmp', encoder_level='char',
                  decoder_level='char')
    src_path = './tmp/src_example.txt'
    pred_path = src_path[:-4] + '_pred.txt'

    print("Predicting test dataset...")
    ckpt_path = get_normalizer_ckpt(normalizer_dir, step=args.normalizer_step)
    print("Load Normalizer model at: ", ckpt_path)
    command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
                   "-beam_size {beam_size} -report_time -gpu 0".format(onmt_path=onmt_package_path,
                                                                       model=ckpt_path,
                                                                       src=src_path,
                                                                       output=pred_path,
                                                                       beam_size=5)
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

    # print result
    result.to_csv(output_file, index=False, header=False)


if __name__ == "__main__":
    main()
