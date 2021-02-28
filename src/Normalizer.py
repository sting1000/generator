import os
import re
import pandas as pd
from tqdm import tqdm
from src.utils import check_folder, read_txt, replace_space, call_rb_API, \
    read_sentence_from_csv, recover_space


class Normalizer:
    def __init__(self, model_yaml_path, prepared_dir, normalizer_dir, onmt_dir='./OpenNMT-py', norm_only=True,
                 encoder_level='char', decoder_level='char', language='en'):
        self.normalizer_dir = normalizer_dir
        self.onmt_dir = onmt_dir
        self.no_classifier = norm_only
        self.encoder_level = encoder_level
        self.decoder_level = decoder_level
        self.prepared_dir = prepared_dir
        self.yaml_path = model_yaml_path
        self.language = language
        if self.yaml_path:
            self.new_yaml_path = '{}/{}'.format(normalizer_dir, model_yaml_path.split('/')[-1])
        check_folder(normalizer_dir + '/checkpoints')
        check_folder(normalizer_dir + '/data')
        check_folder(normalizer_dir + '/tmp')

    def train(self):
        if not self.yaml_path:
            print("This is Rule-based Normalizer. No need to train.")
            return

        # make yaml and data files in normalizer_dir
        self.process_yaml()
        self.process_data()

        # build covab command to use opennmt
        print("Building vocabulary...")
        command_build_vocab = "python {onmt_path}/build_vocab.py " \
                              "-config  {yaml_path} " \
                              "-n_sample -1".format(onmt_path=self.onmt_dir, yaml_path=self.new_yaml_path)
        print(command_build_vocab)
        os.system(command_build_vocab)
        print("Building Done!")

        # train command to use opennmt
        print("Training...")
        command_train = "python {onmt_path}/train.py " \
                        "-config {yaml_path}".format(onmt_path=self.onmt_dir, yaml_path=self.new_yaml_path)
        print(command_train)
        os.system(command_train)
        print("Training Done!")

    def eval(self, key='test', normalizer_step=-1, use_gpu=True):
        """
        default to eval on test dataset
        normalizer_step: steps shown in checkpoints
        """
        tqdm.pandas()
        result = pd.DataFrame()
        result_path = '{}/data/result_{}.csv'.format(self.normalizer_dir, key)

        if not self.yaml_path:
            # Rule based
            data = read_sentence_from_csv("{}/{}.csv".format(self.prepared_dir, key))
            result['src'], result['tgt'] = data['src_token'], data['tgt_token']
            print("Start RB predicting...")
            result['pred'] = data['src_token'].progress_apply(call_rb_API, args=(self.language,))
        else:
            # RNN
            src_path = '{}/data/src_{}.txt'.format(self.normalizer_dir, key)
            pred_path = '{}/data/pred_{}.txt'.format(self.normalizer_dir, key)
            ckpt_path = self.get_ckpt_path(step=normalizer_step)
            print("Predicting {} by {}".format(key, ckpt_path))
            command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
                           "-beam_size {beam_size} -report_time {gpu}".format(onmt_path=self.onmt_dir,
                                                                              model=ckpt_path,
                                                                              src=src_path,
                                                                              output=pred_path,
                                                                              beam_size=5,
                                                                              gpu="-gpu 0" if use_gpu else "")
            print(os.popen(command_pred).read())
            result = self.read_key_txt_files(key)

        # save result
        result.to_csv(result_path, index=False)
        print("Results saved to: ", result_path)

        print("Calculate WER & SER...")
        hyp_path = self.normalizer_dir + '/tmp/eval_hyp.txt'
        ref_path = self.normalizer_dir + '/tmp/eval_ref.txt'
        result[['pred']].to_csv(hyp_path, header=False, index=False)
        result[['tgt']].to_csv(ref_path, header=False, index=False)

        command_wer = 'wer {ref_path} {hyp_path}'.format(ref_path=ref_path, hyp_path=hyp_path)
        print(command_wer)
        print(os.popen(command_wer).read())

    def predict(self, input_path, output_path, normalizer_step=-1, use_gpu=True):
        """
        Use Normalizer to predict input txt file.
        The output txt is saved to output_path
        """
        tqdm.pandas()
        result = pd.DataFrame()
        if not self.yaml_path:
            # Rule based
            result['src'] = read_txt(input_path)
            result['tgt'] = result['src']
            print("Start RB predicting...")
            result['pred'] = result['src'].progress_apply(call_rb_API, args=(self.language,))
        else:
            # RNN
            input_df = pd.DataFrame()
            input_df['src_token'] = read_txt(input_path)
            input_df['src_token'] = input_df['src_token'].str.lower()
            input_df['src_char'] = input_df['src_token'].astype(str).apply(replace_space)
            input_df['tgt_token'] = input_df['src_token']
            input_df['tgt_char'] = input_df['src_char']
            input_df['sentence_id'] = input_df.index

            # make src tgt file
            key = 'tmp'
            src_path = '{}/data/src_{}.txt'.format(self.normalizer_dir, key)
            tgt_path = '{}/data/tgt_{}.txt'.format(self.normalizer_dir, key)
            pred_path = '{}/data/pred_{}.txt'.format(self.normalizer_dir, key)
            input_df[['src_' + self.encoder_level]].to_csv(src_path, header=False, index=False)
            input_df[['tgt_' + self.decoder_level]].to_csv(tgt_path, header=False, index=False)

            # make prediction
            ckpt_path = self.get_ckpt_path(step=normalizer_step)
            print("Predicting test dataset by: ", ckpt_path)
            command_pred = "python {onmt_path}/translate.py -model {model} -src {src} -output {output} " \
                           "-beam_size {beam_size} -report_time {gpu}".format(onmt_path=self.onmt_dir,
                                                                              model=ckpt_path,
                                                                              src=src_path,
                                                                              output=pred_path,
                                                                              beam_size=5,
                                                                              gpu="-gpu 0" if use_gpu else "")
            print(os.popen(command_pred).read())
            result = self.read_key_txt_files(key)

        # print result
        result[['pred']].to_csv(output_path, index=False, header=False)
        print("Prediction saved to: ", output_path)
        return result

    def process_data(self):
        """
        create src and tgt txt data in normalizer_dir/data/
        the data is necessary to run OpenNMT
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
        """
        replace "\{\*PATH\}" in yaml template and make a new one in norm_dir
        """
        with open(self.yaml_path) as f:
            lines = f.readlines()
        with open(self.new_yaml_path, "w+") as f:
            for ind, l in enumerate(lines):
                lines[ind] = re.sub("\{\*PATH\}", str(self.normalizer_dir), l)
            f.writelines(lines)

    def show_ckpt(self):
        """
        print all checkpoint of this normalizer
        """
        _, _, filenames = next(os.walk(self.normalizer_dir + '/checkpoints'))
        print(filenames)

    def read_key_txt_files(self, key):
        """
        read src, tgt, pred file in normalizer_dir
        return a dataframe with 3 columns
        if pred does not exist, use tgt as default
        """
        # define path
        src_path = '{}/data/src_{}.txt'.format(self.normalizer_dir, key)
        tgt_path = '{}/data/tgt_{}.txt'.format(self.normalizer_dir, key)
        pred_path = '{}/data/pred_{}.txt'.format(self.normalizer_dir, key)

        # read files
        result = pd.DataFrame()
        result['src'] = read_txt(src_path)
        result['tgt'] = read_txt(tgt_path)
        result['pred'] = read_txt(pred_path)

        # process format
        if self.encoder_level == 'char':
            result['src'] = result['src'].apply(recover_space)
        if self.decoder_level == 'char':
            result['tgt'] = result['tgt'].apply(recover_space)
            result['pred'] = result['pred'].apply(recover_space)
        return result

    def get_ckpt_path(self, step):
        if step == -1:
            _, _, filenames = next(os.walk(self.normalizer_dir + '/checkpoints'))
            f_name = filenames[-1]
            model_path = self.normalizer_dir + '/checkpoints/{}'.format(f_name)
        else:
            model_path = self.normalizer_dir + '/checkpoints/_step_{}.pt'.format(step)
        return model_path
