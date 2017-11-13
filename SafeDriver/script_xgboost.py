from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import warnings
warnings.filterwarnings("ignore")

from modeling import Modeling as ML
from evaluation import Evaluation as EV

import xgboost as xgb

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', EV.gini(y, pred)
# 100347139332
@ML.metacv
def metacv_xgb(train, valid, test, param):
    watchlist = [(xgb.DMatrix(train.X, train.y), 'train'), 
                (xgb.DMatrix(valid.X, valid.y), 'valid')]
    model = xgb.train(param, 
                        xgb.DMatrix(train.X, train.y), 
                        5000,  
                        watchlist, 
                        feval=gini_xgb, 
                        maximize=True, 
                        verbose_eval=50, 
                        early_stopping_rounds=200)
    return (model.predict(xgb.DMatrix(valid.X), ntree_limit=model.best_ntree_limit+45)
            ,model.predict(xgb.DMatrix(test.X), ntree_limit=model.best_ntree_limit+45))

@ML.gridsearchcv
def gridsearchcv_lgbm(train, valid, param):
    watchlist = [(xgb.DMatrix(train.X, train.y), 'train'), 
                (xgb.DMatrix(valid.X, valid.y), 'valid')]
    model = xgb.train(param, 
                        xgb.DMatrix(train.X, train.y), 
                        5000,  
                        watchlist, 
                        feval=gini_xgb, 
                        maximize=True, 
                        verbose_eval=50, 
                        early_stopping_rounds=200)
    return model.predict(xgb.DMatrix(valid.X), ntree_limit=model.best_ntree_limit+45)   

if __name__ == "__main__":
    mode = "Building..."
    print('['+sys.argv[0].split('/')[-1]+']'+mode)
    path = "./cv/cv_"
    cv = ML.loadcv(path)
    if mode == "Grid Searching...":
        params = {'eta': 0.02, 
                'max_depth': [4], 
                'subsample': 0.9, 
                'colsample_bytree': 0.9, 
                'objective': 'binary:logistic', 
                'eval_metric': 'auc', 
                'seed': 99, 
                'silent': True
               }    
        params = ML.makeparams(params)
        gridsearchcv_xgb("xgboost", cv=cv, params=params, eval_func=EV.gini)
    elif mode == "Building...":
        param = {'eta': 0.02, 
                'max_depth': 4, 
                'subsample': 0.9, 
                'colsample_bytree': 0.9, 
                'objective': 'binary:logistic', 
                'eval_metric': 'auc', 
                'seed': 99, 
                'silent': True
               }  
        test = ML.loadtest(path)
        metacv_xgb("xgboost", cv = cv, test = test, param = param, eval_func = EV.gini)
    else:
        print("Wrong command")