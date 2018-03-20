import os.path
import pandas as pd
import numpy as np

from extractor import Extractor


class Util:

    def __init__(self):
        self.extractor = Extractor('sanderson')

    def save(self, features):
        df = pd.DataFrame(features)
        df.to_csv('output/feature.csv')

    def read(self):
        return pd.read_csv('output/feature.csv')

    def build_row(self, path):
        filename = path.split('/')[2]
        r = filename[filename.find('s')+1:filename.rfind('_')]
        result = int(r)

        #arr = extractor.mfcc(path)
        arr = self.extractor.feature_lib(path)
        arr = [np.insert(a, 0, int(result), axis=0) for a in arr]
        return np.array(arr)

    def buid_dataset(self, path):
        files = next(os.walk(path))[2]
        files = list(files)
        features = []
        for file in files:
            feat = self.build_row(path + file)
            [features.append(f) for f in feat]
        return np.array(features)

    def build_test_dataset(self):
        return self.buid_dataset('samples/test/')

    def build_train_dataset(self):
        return self.buid_dataset('samples/train/')
