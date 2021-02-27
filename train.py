import argparse
from src.Pipeline import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_dir", default='./output/pipeline/distilbert-base_LSTM', type=str,
                        required=False, help="Trained pipeline dir including pipeline_args.txt")
    parser.add_argument("--mode", default='pipeline', type=str,
                        required=False, help="[pipeline, normalizer, classifier]")

    parser.add_argument("--num_train_epochs", default=1, type=int, required=False,
                        help="classifier TrainingArguments")
    parser.add_argument("--per_device_train_batch_size", default=16, type=int, required=False,
                        help="classifier TrainingArguments")
    parser.add_argument("--per_device_eval_batch_size", default=16, type=int, required=False,
                        help="classifier TrainingArguments")
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False,
                        help="classifier TrainingArguments")
    parser.add_argument("--weight_decay", default=1e-2, type=float, required=False,
                        help="classifier TrainingArguments")

    args = parser.parse_args()
    pipeline = load_pipeline(args.pipeline_dir)
    pipeline.train(num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate,
                   weight_decay=args.weight_decay, per_device_train_batch_size=args.per_device_train_batch_size,
                   per_device_eval_batch_size=args.per_device_eval_batch_size, mode=args.mode)


if __name__ == "__main__":
    main()
