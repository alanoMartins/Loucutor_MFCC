from sklearn import mixture
from sklearn.externals import joblib
import numpy as np
import os.path
from sklearn import clone


class GMM_UBM:

    def __init__(self, n_gaussians = 4):
        self.factor = 1 /43
        self.gaussians = n_gaussians
        self.models = []

    def fit(self, train):
        self.ubm = mixture.BayesianGaussianMixture(n_components=self.gaussians, covariance_type='diag', init_params="kmeans")
        full_train = []
        [full_train.append(t) for tr in train for t in tr]
        self.ubm = self.ubm.fit(full_train)

        for idx in range(0, len(train)):
            print("Train model {}".format(idx))
            model = self.max_aposteriori(self.ubm, train[idx])
            print("Model trained")
            self.models.append(model)

    def _update_weigth(self, weigth, ubm_weigth):
        alpha = self.alpha
        return (alpha * weigth + (1 - alpha) * ubm_weigth) * self.factor

    def _update_mean(self, mean, ubm_mean):
        alpha = self.alpha
        return alpha * mean + (1 - alpha) * ubm_mean

    def _update_covariance(self, client_covariance, client_mean, ubm_covariance, ubm_mean, mean):
        alpha = self.alpha
        outer1 = client_mean * client_mean.T
        client_part = client_covariance + outer1
        ubm_part = ubm_covariance + (ubm_mean * ubm_mean.T)
        mm = mean * mean.T
        return (alpha * client_part + (1 - alpha) * ubm_part) * mm

    def _get_alpha(self, data, ubm):
        ll = ubm.score(data)
        return ll / (ll + self.factor)

    def max_aposteriori(self, ubm, data):
        client = clone(ubm)
        client.fit(data)

        self.alpha = self._get_alpha(data, client)

        means = [self._update_mean(client.means_[gauss_idx], ubm.means_[gauss_idx]) for gauss_idx in range(0, self.gaussians)]
        weight = [self._update_weigth(client.weights_[gauss_idx], ubm.weights_[gauss_idx]) for gauss_idx in range(0, self.gaussians)]
        covariance = [self._update_covariance(client.covariances_[gauss_idx], client.means_[gauss_idx], ubm.covariances_[gauss_idx], ubm.means_[gauss_idx], means[gauss_idx]) for gauss_idx in range(0, self.gaussians)]

        client.weights_ = np.array(weight)
        client.means_ = np.array(means)
        client.covariances_ = np.array(covariance)

        return client

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
        predicts_with_impostor = []
        for idx in range(0, len(X_test)):
            test_dataset = X_test[idx]
            best_prob = float("-inf")
            best_class = -1
            for midx in range(0, len(self.models)):
                prob = self.models[midx].score(test_dataset)
                if prob > best_prob:
                    best_prob = prob
                    best_class = midx + 1
            if self.ubm.score(test_dataset) > best_prob:
                predicts_with_impostor.append(-1)
            else:
                predicts_with_impostor.append(best_class)
            predicts.append(best_class)
        return predicts, predicts_with_impostor