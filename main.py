from dataset_util import Util
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from gmm import GMM
import matplotlib.pyplot as plt


def group_X_y(X, y):

    X_by_class = []
    Y_by_class = []
    for cl in np.unique(y):
        values_class = []
        idxs = np.where(y == cl)[0]
        for idx in idxs:
            values_class.append(X[idx])
        Y_by_class.append(y[idxs[0]])
        X_by_class.append(values_class)

    return X_by_class, Y_by_class


if __name__ == '__main__':

    print("Init...")

    print("Convert data")
    util = Util()
    train_dataset = util.build_train_dataset()
    test_dataset = util.build_test_dataset()

    traind_df = pd.DataFrame(train_dataset)

    traind_df.to_json("dataset/train.json")

    traind_df.fillna(0, inplace=True)
    y_train = traind_df.iloc[:, 0].values
    X_train = traind_df.iloc[:, 1:].values

    test_df = pd.DataFrame(test_dataset)

    traind_df.to_json("dataset/test.json")

    test_df.fillna(0, inplace=True)
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values

    X_train, y_train = group_X_y(X_train, y_train)
    X_test, y_test = group_X_y(X_test, y_test)

    print("Creating models")

    n_gauss_test = [2, 4, 6, 8, 12, 16, 20,24, 28, 32, 36, 40]

    predicts = []
    for gauss in n_gauss_test:
        classifier = GMM(n_gaussians=gauss)
        classifier.fit(X_train)
        #classifier.load_models()
        #classifier.persiste_model()
        y_pred = classifier.predict(X_test)
        predicts.append(y_pred)


    print("Generating metrics")

    accuracies = map(lambda x: accuracy_score(y_test, x), predicts)
    accuracies = list(accuracies)

    #f1_scores = map(lambda x: f1_score(y_test, x), predicts)
    #f1_scores = list(f1_scores)


    print("Plot results")

    plt.xlabel('Number of gaussians')
    plt.ylabel('Accuracy')

    y_axis_acc = np.array(accuracies)
    #y_axis_f1 = np.array(f1_scores)
    x_axis = np.array(n_gauss_test)

    plt.plot(x_axis, y_axis_acc)
    #plt.plot(x_axis, y_axis_f1)

    #plt.legend("Accuracy", "F1 Score")
    plt.show()
