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

import xgboost as xgb

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', EV.gini(y, pred)

@ML.metacv
def metacv_xgb(train, valid, test):
    params = {'eta': 0.02, 
              'max_depth': 4, 
              'subsample': 0.9, 
              'colsample_bytree': 0.9, 
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': 99, 
              'silent': True
             }    
    watchlist = [(xgb.DMatrix(train.X, train.y), 'train'), 
                (xgb.DMatrix(valid.X, valid.y), 'valid')]
    model = xgb.train(params, 
                        xgb.DMatrix(train.X, train.y), 
                        5000,  
                        watchlist, 
                        feval=gini_xgb, 
                        maximize=True, 
                        verbose_eval=50, 
                        early_stopping_rounds=200)
    return (model.predict(xgb.DMatrix(valid.X), ntree_limit=model.best_ntree_limit+45)
            ,model.predict(xgb.DMatrix(test.X), ntree_limit=model.best_ntree_limit+45))

if __name__ == "__main__":
    path = "./cv/cv_"
    cv = ML.loadcv(path)
    test = ML.loadtest(path)

    metacv_xgb("xgboost", cv = cv, eval_func = EV.gini)
