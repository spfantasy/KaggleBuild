from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


class dataset:
    def __init__(self, data=None, label=None):
        self.X = data
        self.y = label


class Modeling(object):
    @staticmethod
    def loadtrain(path):
        train = dataset(pd.read_csv(path + "train_X" + ".csv"),
                        pd.read_csv(path + "train_y" + ".csv"))
        return train

    @staticmethod
    def loadtest(path):
        test = dataset(pd.read_csv(path + "test_X" + ".csv"),
                       pd.read_csv(path + "test_y" + ".csv"))
        return test

    @staticmethod
    def loadcv(path):
        cv = []
        i = 0
        while True:
            try:
                train = dataset(pd.read_csv(path + "train_X_" + str(i) + ".csv"),
                                pd.read_csv(path + "train_y_" + str(i) + ".csv"))
                valid = dataset(pd.read_csv(path + "valid_X_" + str(i) + ".csv"),
                                pd.read_csv(path + "valid_y_" + str(i) + ".csv"))
                cv.append([train, valid])
                print("cross validation set %d loaded" % i)
                i += 1
            except:
                return cv

    # I'm a decorator
    @staticmethod
    def metacv(func):
        def wrapper(methodname, cv, test, eval_func):
            testresult = []
            cvresult = []
            cvlabels = []
            for i, data in enumerate(cv):
                print("training CV round %d" % i)
                v, t = func(data[0], data[1], test)
                cvresult.append(v)
                cvlabels.append(data[1].y["target"].as_matrix())
                testresult.append(t)
            # saving testing results
            test.y['target'] = np.mean(testresult, axis=0)
            test.y.to_csv('./predictions/meta_test/%s.csv' %
                          methodname, index=False, float_format='%.5f')
            # saving validation results as stage1result
            stage1result = pd.DataFrame()
            stage1labels = pd.DataFrame()
            stage1result[methodname] = [
                i for validation_split in cvresult for i in validation_split]
            stage1labels["target"] = [
                i for validation_split in cvlabels for i in validation_split]
            score = eval_func(stage1labels.as_matrix(),
                              stage1result.as_matrix())
            print("[metacv@%s] cross validation score = %.4f" %
                  (methodname, score))
        return wrapper

    # I'm a decorator
    @staticmethod
    def gridsearchcv(func):
        def wrapper(methodname, cv, params, eval_func):
            scores = []
            cvresult = []
            cvlabels = []
            for idx, param in params:
                print("testing parameter case %d/%d" % (idx + 1, len(params)))
                for i, data in enumerate(cv):
                    v = func(data[0], data[1], param)
                    cvresult.append(v)
                    cvlabels.append(data[1].y["target"].as_matrix())
                # saving validation results as stage1result
                stage1result = [
                    i for validation_split in cvresult for i in validation_split]
                stage1labels = [
                    i for validation_split in cvlabels for i in validation_split]
                score = eval_func(stage1labels,
                                  stage1result)
                scores.append(score)
            bestidx = scores.index(max(scores))
            print("[gridsearchcv@%s] best cv score = %.4f" %
                  (methodname, scores[bestidx]))
            print("best params are:")
            for key, val in params[bestidx].items():
                print("%10s : %s" % (key, str(val)))
        return wrapper

    # because the return cv predictions are shuffled under validation sets
    # In X->predictions->y
    # stage2 preditions->y
    # y need to be reorder as by cv[i][1].y(and index killed)
    @staticmethod
    def stacking():
        pass
