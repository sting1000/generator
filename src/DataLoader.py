import random
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataLoader:
    def __init__(self, df, test_ratio, num_features, space_letter=0, boundary_letter='-1', window_size=1,
                 random_state=2021):
        self.df = df
        self.space_letter = space_letter
        self.window_size = window_size
        self.boundary_letter = boundary_letter
        self.num_features = num_features

        random.seed(random_state)
        sentence_id_list = list(range(max(df['sentence_id'])))
        random.shuffle(sentence_id_list)
        sep_position = int(test_ratio * len(sentence_id_list))
        test_id = sentence_id_list[:sep_position]
        train_id = sentence_id_list[sep_position:]
        self.test = df[df['sentence_id'].isin(test_id)]
        self.train = df[df['sentence_id'].isin(train_id)]
        self.labels = pd.factorize(df['tag'])[1]

    def save_to_csv(self, path, index=False):
        self.train.to_csv(path + "/train.csv", index=index)
        self.test.to_csv(path + "/test.csv", index=index)

    def get_train_xy(self):
        x_data = self.train['token'].apply(self.__str2vector).to_list()
        x_data = np.array(self.__context_window_transform(x_data))
        y_data = np.array(pd.factorize(self.train['tag'])[0])
        return x_data, y_data

    def get_test_xy(self):
        x_data = self.test['token'].apply(self.__str2vector).to_list()
        x_data = np.array(self.__context_window_transform(x_data))
        y_data = np.array(pd.factorize(self.test['tag'])[0])
        return x_data, y_data

    def get_label(self):
        return self.labels

    def __context_window_transform(self, x_data):
        neo_data = []
        for window_start in tqdm(np.arange(len(x_data) - self.window_size + 1)):
            row = []
            for x in x_data[window_start: window_start + self.window_size]:
                row.append([self.boundary_letter])
                row.append(x)
            row.append([self.boundary_letter])
            neo_data.append([int(x) for y in row for x in y])
        return np.array(neo_data)

    # TODO: class for word embedding
    def __str2vector(self, value):
        vec = np.ones(self.num_features, dtype=int) * self.space_letter
        for ch, ind in zip(list(str(value)), np.arange(self.num_features)):
            vec[ind] = ord(ch)
        return vec
