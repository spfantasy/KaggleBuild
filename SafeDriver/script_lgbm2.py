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

from lightgbm import LGBMClassifier as LGBM

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', EV.gini(y, pred)

@ML.metacv
def metacv_lgbm(train, valid, test):
    params = {
            'learning_rate' : 0.02,
            'n_estimators' : 1090,
            'subsample' : 0.7,
            'subsample_freq' : 2,
            'num_leaves' : 16,
            'seed' : 99,
            }

    model = LGBM(**params)
    model.fit(train.X, train.y)
    return (model.predict_proba(valid.X)[:,1]
            ,model.predict_proba(test.X)[:,1])

if __name__ == "__main__":
    path = "./cv/cv_"
    cv = ML.loadcv(path)
    test = ML.loadtest(path)

    metacv_lgbm("lightgbm2", cv = cv, test = test, eval_func = EV.gini)
