import pandas as pd
from datasets import load_metric, Sequence, ClassLabel, Dataset
import numpy as np
from datasets import DatasetDict
from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm
from src.classification_report import confusion_matrix
from src.utils import read_txt, check_folder




def read_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path, converters={'token': str, 'written': str, 'spoken': str})
    feature_tag = Sequence(ClassLabel(num_classes=3, names=list(pd.factorize(df['tag'])[1])))
    df['tag'] = df['tag'].apply(feature_tag.feature.str2int)
    df_text = df.groupby(['sentence_id']).agg({'token': list, 'tag': list})
    dataset = Dataset.from_pandas(df_text)
    dataset.features["tag"] = feature_tag
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


def save_classifier_result(dataset, tag, output_file):
    """
    return a df['sentence_id', 'token', 'tag']
    token can be a phrase
    tag contains B and O only, B means the token need to be normalized
    """
    tqdm.pandas()
    result = pd.DataFrame(dataset)[['sentence_id', 'token']]
    result['tag'] = tag

    # merge tag and token
    result['token'] = result.apply(merge_col_from_tag, args=('token', 'tag'), axis=1)
    result['tag'] = result.apply(merge_col_from_tag, args=('tag', 'tag'), axis=1)
    result = result.set_index(['sentence_id']).progress_apply(pd.Series.explode).reset_index()
    result.to_csv(output_file, index=False)
    print("Result saved to ", output_file)
    return result


def dataset_to_df(dataset):
    df = pd.DataFrame(dataset)
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    return df


class Classifier:
    def __init__(self, pretrained, prepared_dir, classifier_dir):
        """
        pretrained is None means disable classifier
        """
        self.pretrained = pretrained
        self.classifier_dir = classifier_dir
        self.prepared_dir = prepared_dir
        self.datasets = DatasetDict({
            'train': read_dataset_from_csv(prepared_dir + '/train.csv'),
            'test': read_dataset_from_csv(prepared_dir + '/test.csv'),
            'validation': read_dataset_from_csv(prepared_dir + '/validation.csv')
        })
        self.metric = load_metric("seqeval")
        self.label_list = self.datasets["train"].features["tag"].feature.names
        check_folder(self.classifier_dir)

        if pretrained:
            self.model = AutoModelForTokenClassification.from_pretrained(self.pretrained,
                                                                         num_labels=len(self.label_list))
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
            self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

    def train(self, num_train_epochs=10, learning_rate=1e-5, weight_decay=1e-2,
              per_device_train_batch_size=16, per_device_eval_batch_size=16):
        def compute_metrics(p):
            predictions, labels = p
            true_labels, true_predictions = self.process_pred_labels(predictions, labels)
            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        if self.pretrained:
            tokenized_datasets = self.datasets.map(self.tokenize_and_align_labels, batched=True)
            train_args = TrainingArguments(
                "{}/exp".format(self.classifier_dir),
                evaluation_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                self.model,
                train_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics
            )
            # fine tuning model
            trainer.train()
            trainer.save_model(self.classifier_dir)
            print("Trainer is saved to ", self.classifier_dir)
        else:
            print("No need to train!")

    def eval(self, key='test'):
        tqdm.pandas()
        pred_out_path = '{}/{}_classified_pred.csv'.format(self.classifier_dir, key)
        label_out_path = '{}/{}_classified_label.csv'.format(self.classifier_dir, key)

        print("Predicting {}...".format(key))
        if self.pretrained:
            tokenized_datasets = self.datasets.map(self.tokenize_and_align_labels, batched=True)
            trainer = Trainer(model=self.model, tokenizer=self.tokenizer, data_collator=self.data_collator)
            true_labels, true_predictions = self.predict_dataset(trainer, tokenized_datasets[key])

            pred_result = save_classifier_result(self.datasets[key], true_predictions, pred_out_path)
            label_result = save_classifier_result(self.datasets[key], true_labels, label_out_path)

            # print result
            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            print(" precision:\t{}\n recall:\t{}\n f1:\t{}\n accuracy:\t{}\n".format(
                results["overall_precision"],
                results["overall_recall"],
                results["overall_f1"],
                results["overall_accuracy"])
            )
            confusion_matrix(true_predictions, true_labels)
            return pred_result, label_result
        else:
            df = pd.read_csv('{}/{}.csv'.format(self.prepared_dir, key),
                             converters={'token': str, 'written': str, 'spoken': str})
            result = df[['sentence_id', 'token']].groupby(['sentence_id']).agg({'token': ' '.join})
            result['tag'] = 'B'
            result.to_csv(pred_out_path, index=False)
            result.to_csv(label_out_path, index=False)
            print("Result saved to ", pred_out_path)
            print("Result saved to ", label_out_path)
            return result, result

    def predict(self, input_path, output_path):
        key = 'tmp'
        input_df = pd.DataFrame()


        if self.pretrained:
            input_df['src_token'] = read_txt(input_path)
            input_df['src_token'] = input_df['src_token'].str.lower()
            input_df['token'] = input_df['src_token'].str.split()
            input_df['tag'] = input_df['token'].apply(lambda x: ['O'] * len(x))
            input_df['sentence_id'] = input_df.index

            trainer = Trainer(model=self.model, tokenizer=self.tokenizer, data_collator=self.data_collator)
            feature_tag = Sequence(ClassLabel(num_classes=3, names=self.label_list))
            input_df['tag'] = input_df['tag'].apply(feature_tag.feature.str2int)
            eval_dataset = Dataset.from_pandas(input_df)
            eval_dataset.features["tag"] = feature_tag
            # predict
            tokenized_datasets = DatasetDict({key: eval_dataset}).map(self.tokenize_and_align_labels, batched=True)
            _, true_predictions = self.predict_dataset(trainer, tokenized_datasets[key])
            result = save_classifier_result(eval_dataset, true_predictions, output_path)
            return result
        else:
            input_df['token'] = read_txt(input_path)
            input_df['sentence_id'] = input_df.index
            input_df['tag'] = 'B'
            input_df.to_csv(output_path, index=False)
            print("Result saved to ", output_path)
            return input_df

    def tokenize_and_align_labels(self, examples):
        labels = []
        label_all_tokens = False
        tokenized_inputs = self.tokenizer(examples["token"], truncation=True, is_split_into_words=True)

        for i, label in enumerate(examples["tag"]):
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

    def predict_dataset(self, trainer, data):
        predictions, labels, _ = trainer.predict(data)
        return self.process_pred_labels(predictions, labels)

    def process_pred_labels(self, predictions, labels):
        # Remove ignored index (special tokens)
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions
