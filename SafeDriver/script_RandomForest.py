from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from modeling import dataset
from preprocessing import Preprocessing as PP
from modeling import Modeling as ML
from evaluation import Evaluation as EV

from sklearn.ensemble import RandomForestClassifier as RF


@ML.metacv
def metacv_rf(train, valid, test):
    params = {'n_estimators': 200,
                'criterion': 'gini',
                'max_features': 'log2',
                'max_depth': 6,
                'min_samples_split': 70,
                'min_samples_leaf': 30,
                'n_jobs' : -1,
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
    model = RF(**params)
    model.fit(train.X, train.y)
    return (model.predict(valid.X), model.predict(test.X))

@ML.gridsearchcv
def gridsearchcv_rf(train, valid, param):
    model = RF(**params)
    model.fit(train.X, train.y)
    return model.predict(valid.X)    

if __name__ == "__main__":
    mode = "Grid Searching..."
    print(mode)

    if mode == "Grid Searching..."
        params = {'n_estimators': [25, 100, 200, 500, 1000],
            'criterion': 'gini',
            'max_features': 'log2',
            'max_depth': 6,
            'min_samples_split': 70,
            'min_samples_leaf': 30,
            'n_jobs' : -1,
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
        params = ML.makeparams(params)
        gridsearchcv_rf("randomforest", cv=cv, params=params, eval_func=EV.gini)
    elif mode == "Building"
        path = "./cv/cv_"
        cv = ML.loadcv(path)
        test = ML.loadtest(path)

        metacv_rf("randomforest", cv=cv, test=test, eval_func=EV.gini)
