from sklearn import mixture
from sklearn.externals import joblib
import numpy as np

class GMM:

    def __init__(self, n_gaussians = 4):
        self.gaussians = n_gaussians
        self.models = []

    def fit(self, X_train):
        models = []
        for idx in range(0, len(X_train)):
            model = mixture.GMM(n_components=self.gaussians, covariance_type='diag')
            print("Train model")
            model.fit(X_train[idx])
            print("Model trained")
            self.models.append(model)
            models.append(model)
        return models

    def persiste_model(self):
        for idx in range(0, len(self.models)):
            joblib.dump(self.models[idx], "models/gaussians{}/model_{}.pkl".format(self.gaussians, idx))

    def load_models(self, num_models=41):
        for label in range(0, num_models):
            yield joblib.load("models/model_{}.pkl".format(label))

    def predict(self, X_test):
        predicts = []
        for idx in range(0, len(X_test)):
            test_dataset = X_test[idx]
            best_prob = float("-inf")
            best_class = -1
            for midx in range(0, len(self.models)):
                prob = self.models[midx].score_samples(test_dataset)[0]
                prob = np.sum(prob)
                if prob > best_prob:
                    best_prob = prob
                    best_class = midx + 1
            predicts.append(best_class)
        return predicts