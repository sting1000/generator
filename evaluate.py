import argparse
from src.Pipeline import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_dir", default='./output/pipeline/distilbert-base_LSTM', type=str,
                        required=False, help="Trained pipeline dir including pipeline_args.txt")
    parser.add_argument("--normalize_step", default=-1, type=int, required=False,
                        help="the training step of normalizer, -1 as the last one")
    parser.add_argument("--use_gpu", default=1, type=int, required=False,
                        help="use gpu (1) or not (0) ")
    parser.add_argument("--key", default='test', type=int, required=False,
                        help="which dataset to evaluate")

    args = parser.parse_args()

    pipeline = load_pipeline(args.pipeline_dir)
    pipeline.eval(key=args.key,
                  normalizer_step=args.normalize_step,
                  use_gpu=args.use_gpu)


if __name__ == "__main__":
    main()
