from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import warnings
warnings.filterwarnings("ignore")

from modeling import Modeling as ML
from evaluation import Evaluation as EV

from sklearn.ensemble import RandomForestClassifier as RF


@ML.metacv
def metacv_rf(train, valid, test, param):
    model = RF(**param)
    model.fit(train.X, train.y)
    return (model.predict_proba(valid.X)[:, 1], model.predict_proba(test.X)[:, 1])


@ML.gridsearchcv
def gridsearchcv_rf(train, valid, param):
    model = RF(**param)
    model.fit(train.X, train.y)
    return model.predict_proba(valid.X)[:, 1]


if __name__ == "__main__":
    mode = "Grid Searching..."
    print('[' + sys.argv[0].split('/')[-1] + ']' + mode)
    path = "./cv/cv_"
    cv = ML.loadcv(path)
    if mode == "Grid Searching...":
        params = {'n_estimators': [200, 250, 300],
                  'criterion': 'gini',
                  'max_features': 'log2',
                  'max_depth': 9,  # [6,9,15],
                  'min_samples_split': 75,  # [60,75,100],
                  'min_samples_leaf': 21,  # [15,21,35],
                  'n_jobs': -1,
                  'bootstrap': True,
                  'oob_score': True,
                  'random_state': 99,
                  'verbose': 0,
                  'warm_start': True,
                  'class_weight': {0: 0.0364, 1: 0.9635},
                  }
        params = ML.makeparams(params)
        gridsearchcv_rf("randomforest", cv=cv,
                        params=params, eval_func=EV.gini)
    elif mode == "Building...":
        param = {'n_estimators': 150,
                 'criterion': 'gini',
                 'max_features': 'log2',
                 'max_depth': 9,
                 'min_samples_split': 70,
                 'min_samples_leaf': 30,
                 'n_jobs': -1,
                 # 'min_weight_fraction_leaf': 0.,
                 'max_leaf_nodes': None,
                 # 'min_impurity_decrease': 0.,
                 'bootstrap': True,
                 'oob_score': True,
                 'random_state': 99,
                 'verbose': 0,
                 'warm_start': True,
                 'class_weight': {0: 0.0364, 1: 0.9635},
                 }
        test = ML.loadtest(path)
        metacv_rf("randomforest", cv=cv, test=test,
                  param=param, eval_func=EV.gini)
    else:
        print("Wrong command")
