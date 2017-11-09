from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from copy import deepcopy


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
                print("cross validation set %d loaded" %(i+1))
                i += 1
            except:
                return cv

    # I'm a decorator
    @staticmethod
    def metacv(func):
        def wrapper(methodname, cv, test, param, eval_func):
            testresult = []
            cvresult = []
            cvlabels = []
            for i, data in enumerate(cv):
                print("training CV round %d" % (i+1))
                #core wrapped
                v, t = func(data[0], data[1], test, param)
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
            stage1result.to_csv('./predictions/meta_train/%s.csv' %
                          methodname, index=False, float_format='%.5f')
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
            for idx, param in enumerate(params):
                print("testing parameter case %d/%d" % (idx + 1, len(params)))
                for i, data in enumerate(cv):
                    #core wrapped
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
            print('{')
            for key, val in params[bestidx].items():
                print("\t'%s' : %s," % (key, str(val)))
            print('}')
        return wrapper

    @staticmethod
    def makeparams(params):
        keys = list(params)
        ans = []
        def dfs(remain, path):
            if len(remain) == 0:
                ans.append(path)
            else:
                key = remain[0]
                choices = params[key]
                if isinstance(choices, list):
                    for choice in choices:
                        thispath = deepcopy(path)
                        if choice is not None:
                            thispath[key] = choice
                        dfs(remain[1:], thispath)
                else:
                    path[key] = choices
                    dfs(remain[1:], path)
            return

        dfs(keys, {})
        return ans

    # because the return cv predictions are shuffled under validation sets
    # In X->predictions->y
    # stage2 preditions->y
    # y need to be reorder as by cv[i][1].y(and index killed)
    @staticmethod
    def stacking():
        pass
