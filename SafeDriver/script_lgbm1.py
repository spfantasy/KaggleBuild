from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import warnings
warnings.filterwarnings("ignore")

from modeling import Modeling as ML
from evaluation import Evaluation as EV

from lightgbm import LGBMClassifier as LGBM


@ML.metacv
def metacv_lgbm(train, valid, test, param):
    model = LGBM(**param)
    model.fit(train.X, train.y)
    return (model.predict_proba(valid.X)[:,1]
            ,model.predict_proba(test.X)[:,1])

@ML.gridsearchcv
def gridsearchcv_lgbm(train, valid, param):
    model = LGBM(**param)
    model.fit(train.X, train.y)
    return model.predict_proba(valid.X)[:,1]

if __name__ == "__main__":
    mode = "Building..."
    
    print('['+sys.argv[0].split('/')[-1]+']'+mode)
    path = "./cv/cv_"
    cv = ML.loadcv(path)
    if mode == "Grid Searching...":
        params = {
            'learning_rate' : 0.02,
            'n_estimators' : [650],
            'max_bin' : 10,
            'subsample' : 0.8,
            'subsample_freq' : 10,
            'colsample_bytree' : 0.8,
            'min_child_samples' : 500,
            'seed' : 99,
            }
        params = ML.makeparams(params)
        gridsearchcv_lgbm("lightgbm1", cv=cv, params=params, eval_func=EV.gini)
    elif mode == "Building...":
        param = {
            'learning_rate' : 0.02,
            'n_estimators' : 650,
            'max_bin' : 10,
            'subsample' : 0.8,
            'subsample_freq' : 10,
            'colsample_bytree' : 0.8,
            'min_child_samples' : 500,
            'seed' : 99,
            }
        test = ML.loadtest(path)
        metacv_lgbm("lightgbm1", cv=cv, test=test, param = param, eval_func=EV.gini)
    else:
        print("Wrong command")