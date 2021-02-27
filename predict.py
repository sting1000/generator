import argparse
from src.Pipeline import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_dir", default='./output/pipeline/distilbert-base_LSTM', type=str,
                        required=False, help="Trained pipeline dir including pipeline_args.txt")
    parser.add_argument("--input_path", default='./example/input.txt', type=str,
                        required=False, help="txt file to be predicted")
    parser.add_argument("--output_path", default='./example/output.txt', type=str,
                        required=False, help="output txt file path")
    parser.add_argument("--normalize_step", default=-1, type=int, required=False,
                        help="the training step of normalizer, -1 as the last one")
    parser.add_argument("--use_gpu", default=1, type=int, required=False,
                        help="use gpu (1) or not (0) ")

    args = parser.parse_args()

    pipeline = load_pipeline(args.pipeline_dir)
    pipeline.predict(input_path=args.input_path,
                     output_path=args.output_path,
                     normalize_step=args.normalize_step,
                     use_gpu=False)


if __name__ == "__main__":
    main()
