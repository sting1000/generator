import argparse
import pandas as pd
from tqdm import tqdm
from eval_pipeline import call_rb_API
from src.utils import check_folder, clean_string


# TODO: test on eval_pipeline.py, and remove this file
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appendix", default=None, type=str, required=True,
                        help="normalizer_dir")
    parser.add_argument("--language", default='de', type=str, required=False,
                        help="language")

    args = parser.parse_args()
    appendix = args.appendix
    language = args.language
    classified_path = './output/test_classified_pred_{}.csv'.format(appendix)
    classified_df = pd.read_csv(classified_path)
    data = classified_df[classified_df.tag != 'O'].astype(str)

    tqdm.pandas()
    data['src'] = data['token']
    data['pred'] = data['token'].progress_apply(call_rb_API, args=(language,))
    data[['pred']].to_csv(classified_path[:-4] + '.txt', index=False, header=False)

    classified_df['pred'] = classified_df['token'].astype(str)
    id_TBNorm = classified_df.index[classified_df['tag'] == 'B'].tolist()
    classified_df.loc[id_TBNorm, 'pred'] = data['pred'].tolist()
    result = classified_df.groupby(['sentence_id']).agg({'pred': ' '.join})

    # add label and src to result
    test_path = './output/test.csv'
    test = pd.read_csv(test_path)
    test = test[['sentence_id', 'token_id', 'language', 'written', 'spoken']].drop_duplicates()
    test['src'] = test['spoken'].astype(str)
    test['label'] = test['written'].astype(str)
    test = test.groupby(['sentence_id']).agg({'src': ' '.join, 'label': ' '.join})
    result['tgt'] = test['label'].apply(clean_string)
    result['src'] = test['src']
    result['pred'] = result['pred'].apply(clean_string)
    # print result
    check_folder('./rb_pipeline_result_test')

    correct_num = sum(result['pred'] == result['tgt'])
    print("Pipeline Error: ", len(result) - correct_num)
    print("Pipeline Total: ", len(result))
    print("Pipeline Accuracy: ", correct_num / len(result))
    result.to_csv('./rb_pipeline_result_test/' + classified_path.split('/')[-1])


if __name__ == "__main__":
    main()
