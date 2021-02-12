import pandas as pd
from datasets import load_metric, Sequence, ClassLabel, Dataset
import numpy as np
from datasets import DatasetDict
from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

folder_path = './output'
pretrained_path = 'distilbert-base-german-cased'
save_model_path = './output/classifier'
metric = load_metric("seqeval")
label_all_tokens = False
save_model_path += ('/' + pretrained_path)

args = TrainingArguments(
    f"exp-distilbert",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
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


def read_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path).astype(str)
    feature_tag = Sequence(ClassLabel(num_classes=3, names=list(pd.factorize(df['tag'])[1])))
    df['tag'] = df['tag'].apply(feature_tag.feature.str2int)
    df_text = df.groupby(['sentence_id']).agg({'token': list, 'tag': list})
    dataset = Dataset.from_pandas(df_text)
    dataset.features[f"tag"] = feature_tag
    return dataset


datasets = DatasetDict({
    'train': read_dataset_from_csv(folder_path + '/train.csv'),
    'test': read_dataset_from_csv(folder_path + '/test.csv'),
    'validation': read_dataset_from_csv(folder_path + '/validation.csv')
})
model = AutoModelForTokenClassification.from_pretrained(pretrained_path, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
label_list = datasets["train"].features[f"tag"].feature.names


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["token"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tag"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
label_list = datasets["train"].features[f"tag"].feature.names
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(save_model_path)
print("Trainer is saved to ", save_model_path)


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


def dataset_to_df(dataset):
    df = pd.DataFrame(dataset)
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    return df


def merge_col_from_tag(row, col, tag):
    l = row[col].copy()
    if col in ['tag', 'tag_pred']:
        for tup in get_entities(row[tag]):
            for i in range(tup[1], tup[2] + 1):
                l.pop(tup[1])
            l.insert(tup[1], 'B')
    else:
        for tup in get_entities(row[tag]):
            text = row[col][tup[1]: tup[2] + 1]
            for i in range(tup[1], tup[2] + 1):
                l.pop(tup[1])
            l.insert(tup[1], ' '.join(text))
    return l


def save_result(df, output_file):
    df['token'] = df.apply(merge_col_from_tag, args=('token', 'tag'), axis=1)
    df['tag'] = df.apply(merge_col_from_tag, args=('tag', 'tag'), axis=1)
    # df = df[['sentence_id', 'token', 'tag']]
    df = df.set_index(['sentence_id']).apply(pd.Series.explode).reset_index()
    df.to_csv(output_file, index=False)
    return df


for key in ['train', 'test', 'validation']:
    print("Predicting and Saving {} dataset...".format(key))
    label, pred, results = predict_dataset(trainer, tokenized_datasets[key])
    df = dataset_to_df(datasets[key])
    save_result(df, '{}/{}_classified_label.csv'.format(folder_path, key))

    df_classified = df.copy()
    df_classified['tag'] = pred
    save_result(df_classified, '{}/{}_classified_pred.csv'.format(folder_path, key))

    print("{} results:", key)
    print(results)
