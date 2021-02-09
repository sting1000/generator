import xgboost as xgb


class Classifier:
    def __init__(self, x_train, y_train, x_valid, y_valid):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        # TODO: support more models
        self.model_name = xgb
        self.model = None
        # self.params = params

    def train(self, **kwargs):
        self.model = self.model_name.train(**kwargs)

    def predict(self, **kwargs):
        pred = self.model.predict(**kwargs)
        return pred

    def save_model(self, out_path):
        self.model.save_model(out_path)

    def load_model(self, model_path):
        self.model = xgb.Booster({'nthread': -1})  # init model
        self.model.load_model(model_path)
