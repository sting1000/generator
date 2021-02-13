import argparse
import pandas as pd
from datasets import load_metric, Sequence, ClassLabel, Dataset
import numpy as np
from datasets import DatasetDict
from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification


def read_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path, converters={'token': str, 'written': str, 'spoken': str})
    feature_tag = Sequence(ClassLabel(num_classes=3, names=list(pd.factorize(df['tag'])[1])))
    df['tag'] = df['tag'].apply(feature_tag.feature.str2int)
    df_text = df.groupby(['sentence_id']).agg({'token': list, 'tag': list})
    dataset = Dataset.from_pandas(df_text)
    dataset.features[f"tag"] = feature_tag
    return dataset


def merge_col_from_tag(row, col, tag):
    l = row[col].copy()
    if col in ['tag', 'tag_pred']:
        for tup in get_entities(row[tag])[::-1]:
            for i in range(tup[1], tup[2] + 1):
                l.pop(tup[1])
            l.insert(tup[1], 'B')
    else:
        for tup in get_entities(row[tag])[::-1]:
            text = row[col][tup[1]: tup[2] + 1]
            for i in range(tup[1], tup[2] + 1):
                l.pop(tup[1])
            l.insert(tup[1], ' '.join(text))
    return l


def save_result(df, output_file):
    df['token'] = df.apply(merge_col_from_tag, args=('token', 'tag'), axis=1)
    df['tag'] = df.apply(merge_col_from_tag, args=('tag', 'tag'), axis=1)
    df = df.set_index(['sentence_id']).apply(pd.Series.explode).reset_index()
    df.to_csv(output_file, index=False)
    return df


def dataset_to_df(dataset):
    df = pd.DataFrame(dataset)
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", default='./output', type=str, required=False,
                        help="The output dir from prepare.py default as ./output")
    parser.add_argument("--pretrained", default='distilbert-base-german-cased', type=str, required=False,
                        help="Load model from huggingface pretrained, default as None")
    parser.add_argument("--classifier_dir", default='./output/classifier/distilbert-base-german-cased', type=str, required=False,
                        help="Directory to save model and data")
    parser.add_argument("--num_train_epochs", default=10, type=int, required=False,
                        help="TrainingArguments")
    parser.add_argument("--per_device_train_batch_size", default=16, type=int, required=False,
                        help="TrainingArguments")
    parser.add_argument("--per_device_eval_batch_size", default=16, type=int, required=False,
                        help="TrainingArguments")
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False,
                        help="TrainingArguments")
    parser.add_argument("--weight_decay", default=1e-2, type=float, required=False,
                        help="TrainingArguments")

    # init
    args = parser.parse_args()
    prepared_dir = args.prepared_dir
    pretrained_path = args.pretrained
    save_model_path = args.classifier_dir
    train_args = TrainingArguments(
        "exp-{}".format(args.pretrained),
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

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

    def predict_dataset(trainer, data):
        predictions, labels, _ = trainer.predict(data)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return true_labels, true_predictions, results

    # tokenize and make datasets dict
    datasets = DatasetDict({
        'train': read_dataset_from_csv(prepared_dir + '/train.csv'),
        'test': read_dataset_from_csv(prepared_dir + '/test.csv'),
        'validation': read_dataset_from_csv(prepared_dir + '/validation.csv')
    })
    metric = load_metric("seqeval")
    model = AutoModelForTokenClassification.from_pretrained(pretrained_path, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    label_list = datasets["train"].features[f"tag"].feature.names

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    label_list = datasets["train"].features[f"tag"].feature.names
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # fine tuning model
    trainer.train()
    trainer.save_model(args.classifier_dir)
    print("Trainer is saved to ", args.classifier_dir)

    # save results
    results_list = []
    results_index = ['train', 'test', 'validation']
    for key in results_index:
        print("Predicting and Saving {} dataset...".format(key))
        label, pred, results = predict_dataset(trainer, tokenized_datasets[key])
        df = dataset_to_df(datasets[key])
        df_classified = df.copy()
        df_classified['tag'] = pred
        save_result(df, '{}/{}_classified_label.csv'.format(args.classifier_dir, key))
        save_result(df_classified, '{}/{}_classified_pred.csv'.format(args.classifier_dir, key))

        results_formatted = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "bsize": results['TBNorm']['number'],
            "overall_acc": results["overall_accuracy"]
        }
        results_list.append(results_formatted)
    resutls_classifier = pd.DataFrame(results_list, results_index)
    resutls_classifier.to_csv(args.classifier_dir + '/results_classifier_test.csv', index=False)
    print(resutls_classifier)


if __name__ == "__main__":
    main()
