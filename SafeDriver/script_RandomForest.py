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
    params = {'n_estimators': 30,
              'criterion': 'gini',
              'max_features': 'log2',
              'max_depth': 10,
              'min_samples_split': 2,
              'min_samples_leaf': 1,
              'min_weight_fraction_leaf': 0.,
              'max_leaf_nodes': None,
              'min_impurity_decrease': 0.,
              'bootstrap': True,
              'oob_score': True,
              'random_state': 99,
              'verbose': 0,
              'warm_start': False,
              'class_weight': {1: 0.0364, 0: 0.9635},
              }

    model = RF(**params)
    model.fit(train.X, train.y)
    return (model.predict(valid.X), model.predict(test.X))


if __name__ == "__main__":
    path = "./cv/cv_"
    cv = ML.loadcv(path)
    test = ML.loadtest(path)

    metacv_rf("randomforest", cv=cv, test=test, eval_func=EV.gini)
