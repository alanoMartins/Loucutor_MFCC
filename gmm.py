from sklearn import mixture
from sklearn.externals import joblib
import numpy as np
import os.path

class GMM:

    def __init__(self, n_gaussians = 4):
        self.gaussians = n_gaussians
        self.models = []

    def fit(self, X_train):
        models = []
        impostor = self.create_impostor(X_train)
        X_train.append(list(impostor))

        for idx in range(0, len(X_train)):
            model = mixture.BayesianGaussianMixture(n_components=self.gaussians, covariance_type='diag', init_params="kmeans")
            print("Train model")
            model.fit(X_train[idx])
            print("Model trained")
            self.models.append(model)
            models.append(model)
        return models


    def create_impostor(self, X_train):
        data_flat = []
        for data in X_train:
            data_flat.append(np.array(data).flatten())
        sizes = [len(d) for d in data_flat]
        min_size = np.min(list(sizes))
        data_flat = [d[0:min_size] for d in data_flat]
        s = np.sum(data_flat, axis=0, keepdims=True)
        mean = s / len(X_train)
        couter = mean.shape[1] / 13
        mean = mean.reshape(int(couter), 13)
        return mean

    def persiste_model(self):
        for idx in range(0, len(self.models)):
            joblib.dump(self.models[idx], "models/gaussians{}/model_{}.pkl".format(self.gaussians, idx))

    def load_models(self):
        path = "models/gaussians{}/".format(self.gaussians)
        files = next(os.walk(path))[2]
        files = list(files)
        self.models = []
        for label in files:
            model = joblib.load(path + label)
            self.models.append(model)

    def predict(self, X_test):
        predicts = []
        for idx in range(0, len(X_test)):
            test_dataset = X_test[idx]
            best_prob = float("-inf")
            best_class = -1
            for midx in range(0, len(self.models)):
                prob = self.models[midx].score(test_dataset)
                prob = np.sum(prob)
                if prob > best_prob:
                    best_prob = prob
                    best_class = midx + 1
            predicts.append(best_class)
        return predicts