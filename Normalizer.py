import os
import re
import pandas as pd
from src.utils import check_folder, get_normalizer_ckpt, onmt_txt_to_df, read_txt, replace_space


class Normalizer:
    def __init__(self, model_yaml_path, prepared_dir, normalizer_dir, onmt_dir='./OpenNMT-py', no_classifier=0,
                 encoder_level='char', decoder_level='char'):
        self.normalizer_dir = normalizer_dir
        self.onmt_dir = onmt_dir
        self.no_classifier = no_classifier
        self.encoder_level = encoder_level
        self.decoder_level = decoder_level
        self.prepared_dir = prepared_dir
        self.yaml_path = model_yaml_path
        self.new_yaml_path = '{}/{}'.format(normalizer_dir, model_yaml_path.split('/')[-1])
        check_folder(normalizer_dir + '/checkpoints')
        check_folder(normalizer_dir + '/data')
        check_folder(normalizer_dir + '/tmp')

    def train(self):
        # make yaml and data files in normalizer_dir
        self.process_yaml()
        self.process_data()

        # build covab command to use opennmt
        print("Building vocabulary...")
        command_build_vocab = "python {onmt_path}/build_vocab.py " \
                              "-config  {yaml_path} " \
                              "-n_sample -1".format(onmt_path=self.onmt_dir, yaml_path=self.new_yaml_path)

        # train command to use opennmt
        print("Training...")
        command_train = "python {onmt_path}/train.py " \
                        "-config {yaml_path}".format(onmt_path=self.onmt_dir, yaml_path=self.new_yaml_path)

        # execute opennmt
        os.system(command_build_vocab)
        os.system(command_train)

    def eval(self, key='test', normalizer_step=-1):
        """
        default to eval on test dataset
        """
        src_path = '{}/data/src_{}.txt'.format(self.normalizer_dir, key)
        pred_path = '{}/data/pred_{}.txt'.format(self.normalizer_dir, key)
        result_path = '{}/data/result_{}.csv'.format(self.normalizer_dir, key)
        ckpt_path = get_normalizer_ckpt(self.normalizer_dir, step=normalizer_step)
        print("Predicting {} by {}".format(key, ckpt_path))
        command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} -gpu 0 " \
                       "-beam_size {beam_size} -report_time".format(onmt_path=self.onmt_dir,
                                                                    model=ckpt_path,
                                                                    src=src_path,
                                                                    output=pred_path,
                                                                    beam_size=5)
        os.system(command_pred)
        result = onmt_txt_to_df(self.normalizer_dir, key, self.encoder_level, self.decoder_level)
        result.to_csv(result_path, index=False)
        print("Results saved to: ", result_path)

        print("Calculate WER & SER...")
        hyp_path = self.normalizer_dir + '/tmp/eval_hyp.txt'
        ref_path = self.normalizer_dir + '/tmp/eval_ref.txt'
        result[['pred']].to_csv(hyp_path, header=False, index=False)
        result[['tgt']].to_csv(ref_path, header=False, index=False)

        print('wer -c {ref_path} {hyp_path}'.format(ref_path=ref_path, hyp_path=hyp_path))

    def predict(self, input_path, output_path, tgt_path=None, normalizer_step=-1):
        input_df = pd.DataFrame()
        input_df['src_token'] = read_txt(input_path)
        input_df['src_token'] = input_df['src_token'].str.lower()
        input_df['src_char'] = input_df['src_token'].astype(str).apply(replace_space)

        if not tgt_path:
            input_df['tgt_token'] = input_df['src_token']
            input_df['tgt_char'] = input_df['src_char']
        else:
            input_df['tgt_token'] = read_txt(input_path)
            input_df['tgt_token'] = input_df['tgt_token'].str.lower()
            input_df['tgt_char'] = input_df['tgt_char'].astype(str).apply(replace_space)
        input_df['sentence_id'] = input_df.index

        # make src tgt file
        key = 'tmp'
        src_path = '{}/data/src_{}.txt'.format(self.normalizer_dir, key)
        tgt_path = '{}/data/tgt_{}.txt'.format(self.normalizer_dir, key)
        pred_path = '{}/data/pred_{}.txt'.format(self.normalizer_dir, key)
        result_path = '{}/data/result_{}.csv'.format(self.normalizer_dir, key)
        input_df[['src_' + self.encoder_level]].to_csv(src_path, header=False, index=False)
        input_df[['tgt_' + self.decoder_level]].to_csv(tgt_path, header=False, index=False)

        # make prediction
        ckpt_path = get_normalizer_ckpt(self.normalizer_dir, step=normalizer_step)
        print("Predicting test dataset by: ", ckpt_path)
        command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
                       "-beam_size {beam_size} -report_time -gpu 0".format(onmt_path=self.onmt_dir,
                                                                           model=ckpt_path,
                                                                           src=src_path,
                                                                           output=pred_path,
                                                                           beam_size=5)
        os.system(command_pred)

        result = onmt_txt_to_df(self.normalizer_dir, key, self.encoder_level, self.decoder_level)
        result[['pred']].to_csv(output_path, index=False, header=False)
        print("Prediction saved to: ", output_path)

    def process_data(self):
        """
        create txt data in normalizer_dir/data using prepared_dir files
        """
        for key in ['train', 'validation', 'test']:
            df = pd.read_csv('{}/{}.csv'.format(self.prepared_dir, key),
                             converters={'token': str, 'written': str, 'spoken': str})
            data = df if self.no_classifier else df[df.tag != 'O']  # choose which part as src
            data = data[['sentence_id', 'token_id', 'language', 'written', 'spoken']].drop_duplicates()
            data['tgt_token'], data['src_token'] = data['written'], data['spoken']

            if self.no_classifier:
                data = data.groupby(['sentence_id']).agg({'src_token': ' '.join, 'tgt_token': ' '.join})
            data['tgt_char'] = data['tgt_token'].apply(replace_space)
            data['src_char'] = data['src_token'].apply(replace_space)

            print("Making src tgt for: ", key)
            src_path = '{}/data/src_{}.txt'.format(self.normalizer_dir, key)
            tgt_path = '{}/data/tgt_{}.txt'.format(self.normalizer_dir, key)
            data[['src_' + self.encoder_level]].to_csv(src_path, header=False, index=False)
            data[['tgt_' + self.decoder_level]].to_csv(tgt_path, header=False, index=False)

    def process_yaml(self):
        with open(self.yaml_path) as f:
            lines = f.readlines()
        with open(self.new_yaml_path, "w+") as f:
            for ind, l in enumerate(lines):
                lines[ind] = re.sub("\{\*PATH\}", str(self.normalizer_dir), l)
            f.writelines(lines)

    def show_ckpt(self):
        _, _, filenames = next(os.walk(self.normalizer_dir + '/checkpoints'))
        print(filenames)