from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import warnings
warnings.filterwarnings("ignore")
from os import listdir
from os.path import join

from modeling import Modeling as ML
from evaluation import Evaluation as EV
from modeling import dataset
import pandas as pd

from sklearn.linear_model import LogisticRegression as LR

if __name__ == "__main__":
    path = "./cv/cv_"
    meta_train_path = "./predictions/meta_train/"
    meta_test_path = "./predictions/meta_test/"
    cv = ML.loadcv(path)
    test = ML.loadtest(path)
    # construct training set
    filenamelist = [f[:-4]
                    for f in listdir(meta_train_path) if join(meta_train_path, f).endswith('.csv')]
    filepathlist = [join(meta_train_path, f) for f in listdir(
        meta_train_path) if join(meta_train_path, f).endswith('.csv')]
    metas = [pd.read_csv(mypath)['target'] for mypath in filepathlist]
    df = pd.concat(metas, axis=1)
    df.columns = filenamelist
    train = dataset(df, pd.concat(
        [valid.y for _, valid in cv], axis=0).reset_index(drop=True))
    # construct testing set
    filenamelist = [f[:-4]
                    for f in listdir(meta_test_path) if join(meta_test_path, f).endswith('.csv')]
    filepathlist = [join(meta_test_path, f) for f in listdir(
        meta_test_path) if join(meta_test_path, f).endswith('.csv')]
    metas = [pd.read_csv(mypath)['target'] for mypath in filepathlist]
    df = pd.concat(metas, axis=1)
    df.columns = filenamelist
    test = dataset(df, test.y)
    #####
    ML.stacking(train, test, LR())#class_weight= {0: 0.0364, 1: 0.9635}))
