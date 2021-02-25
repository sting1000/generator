import os
from Classifier import Classifier
from Normalizer import Normalizer
from src.utils import read_sentence_from_csv, check_folder


class Pipeline:
    def __init__(self, pipeline_dir, prepared_dir, classifier_dir, pretrained, normalizer_dir, model_yaml_path,
                 encoder_level='char', decoder_level='char', onmt_dir='./OpenNMT-py'):
        self.pipeline_dir = pipeline_dir
        self.prepared_dir = prepared_dir
        self.classifier_dir = classifier_dir
        self.pretrained = pretrained
        self.normalizer_dir = normalizer_dir
        self.encoder_level = encoder_level
        self.decoder_level = decoder_level
        self.onmt_dir = onmt_dir

        check_folder(self.pipeline_dir)
        check_folder(self.pipeline_dir + '/tmp')
        self.Classifier = Classifier(pretrained, prepared_dir, classifier_dir)
        self.Normalizer = Normalizer(model_yaml_path, prepared_dir, normalizer_dir,
                                     no_classifier=False if pretrained else True,
                                     onmt_dir=onmt_dir,
                                     encoder_level=encoder_level,
                                     decoder_level=decoder_level)

    def train(self, num_train_epochs=10, learning_rate=1e-5, weight_decay=1e-2,
              per_device_train_batch_size=16, per_device_eval_batch_size=16):
        print("Start training classifier...")
        self.Classifier.train(num_train_epochs, learning_rate, weight_decay,
                              per_device_train_batch_size, per_device_eval_batch_size)

        print("Start training normalizer...")
        self.Normalizer.train()

    def eval(self, key='test', normalizer_step=-1):
        # make test data as sentence
        df = read_sentence_from_csv("{}/{}.csv".format(self.prepared_dir, key))
        input_path = '{}/{}_input.txt'.format(self.pipeline_dir, key)
        target_path = '{}/{}_target.txt'.format(self.pipeline_dir, key)
        cls_path = '{}/{}_classified.csv'.format(self.pipeline_dir, key)
        tbn_path = '{}/{}_TBNormed.txt'.format(self.pipeline_dir, key)
        norm_path = '{}/{}_normed.txt'.format(self.pipeline_dir, key)
        output_path = '{}/{}_output.txt'.format(self.pipeline_dir, key)

        df[['src_token']].to_csv(input_path, header=False, index=False)
        df[['tgt_token']].to_csv(target_path, header=False, index=False)

        print("Start evaluating classifier...")
        df_cls = self.Classifier.predict(input_path, cls_path)
        df_TBNorm = df_cls[df_cls.tag != 'O']
        df_TBNorm['token'].to_csv(tbn_path, header=False, index=False)

        print("Start evaluating normalizer...")
        df_nor = self.Normalizer.predict(input_path=tbn_path, output_path=norm_path,
                                         normalizer_step=normalizer_step)

        df_cls['pred'] = df_cls['token']
        id_TBNorm = df_cls.index[df_cls['tag'] == 'B'].tolist()
        df_cls.loc[id_TBNorm, 'pred'] = df_nor['pred'].tolist()
        result = df_cls.groupby(['sentence_id']).agg({'pred': ' '.join})
        result[['pred']].to_csv(output_path, header=False, index=False)

        command_wer = 'wer {ref_path} {hyp_path}'.format(ref_path=target_path, hyp_path=output_path)
        print(command_wer)
        print(os.popen(command_wer).read())

    def predict(self, input_path, output_path, normalizer_step):
        tmp_dir = self.pipeline_dir + '/tmp'
        key = 'tmp'
        cls_path = '{}/{}_classified.csv'.format(tmp_dir, key)
        tbn_path = '{}/{}_TBNormed.txt'.format(tmp_dir, key)
        norm_path = '{}/{}_normed.txt'.format(tmp_dir, key)

        print("Start predicting classifier...")
        df_cls = self.Classifier.predict(input_path, cls_path)
        df_TBNorm = df_cls[df_cls.tag != 'O']
        df_TBNorm['token'].to_csv(tbn_path, header=False, index=False)

        print("Start predicting normalizer...")
        df_nor = self.Normalizer.predict(input_path=tbn_path, output_path=norm_path,
                                         normalizer_step=normalizer_step)

        df_cls['pred'] = df_cls['token']
        id_TBNorm = df_cls.index[df_cls['tag'] == 'B'].tolist()
        df_cls.loc[id_TBNorm, 'pred'] = df_nor['pred'].tolist()
        result = df_cls.groupby(['sentence_id']).agg({'pred': ' '.join})
        result[['pred']].to_csv(output_path, header=False, index=False)
        print("Prediction saved to: ", output_path)
