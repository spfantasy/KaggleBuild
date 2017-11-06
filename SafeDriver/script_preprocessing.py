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

def PreprocessOriginalData(train, test):
    # preprocessing training/testing
    print("preprocessing on whole training/testing set")
    # deepcopy the train/test set to preprocess
    from copy import deepcopy
    train = deepcopy(train)
    test = deepcopy(test)
    test.X = pd.DataFrame.copy(test.X)
    # fillNA
    train.X, NAmethod = PP.fillNA(train.X)
    test.X, _ = PP.fillNA(test.X, NAmethod)
    if True:
        # set down which list need 2b dummied
        mydummylist = PP.dummylist(
            pd.concat([train.X, test.X], axis=0))
        myheader = PP.makehead(
            pd.concat([train.X, test.X], axis=0), mydummylist)
    train.X = PP.dummy(
        PP.addhead(train.X, myheader), mydummylist)
    train.X = PP.rmhead(train.X)
    test.X = PP.dummy(
        PP.addhead(test.X, myheader), mydummylist)
    test.X = PP.rmhead(test.X)
    train.X, STDmethod = PP.standardize(train.X)
    test.X, _ = PP.standardize(test.X, STDmethod)
    print("saving %strain_X.csv (%d row * %d col)"%(filename, train.X.shape[0], train.X.shape[1]))
    train.X.to_csv(filename + "train_X.csv",
                   index=False, float_format="%.5f")
    train.y.to_csv(filename + "train_y.csv",
                   index=False, float_format="%.5f")
    print("saving %stest_X.csv (%d row * %d col)"%(filename, test.X.shape[0], test.X.shape[1]))
    test.X.to_csv(filename + "test_X.csv",
                  index=False, float_format="%.5f")
    test.y.to_csv(filename + "test_y.csv",
                  index=False, float_format="%.5f")
    return mydummylist, myheader

def KFoldsPreprocess(train, test, mydummylist, myheader):
    # do methods in preprocessing in each fold
    for i, (train_idx, valid_idx) in enumerate(folds):
        print("preprocessing on cv #%d" % i)
        train_this_cut = dataset(
            train.X.loc[train_idx], train.y.loc[train_idx])
        valid_this_cut = dataset(
            train.X.loc[valid_idx], train.y.loc[valid_idx])
        train_this_cut.X, NAmethod = PP.fillNA(train_this_cut.X)
        train_this_cut.X = PP.dummy(
            PP.addhead(train_this_cut.X, myheader), mydummylist)
        train_this_cut.X = PP.rmhead(train_this_cut.X)
        train_this_cut.X, STDmethod = PP.standardize(
            train_this_cut.X)
        valid_this_cut.X, _ = PP.fillNA(
            valid_this_cut.X, NAmethod)
        valid_this_cut.X = PP.dummy(
            PP.addhead(valid_this_cut.X, myheader), mydummylist)
        valid_this_cut.X = PP.rmhead(valid_this_cut.X)
        valid_this_cut.X, _ = PP.standardize(
            valid_this_cut.X, STDmethod)

        print("saving %strain_X_%d.csv (%d row * %d col)"%(filename, i, train_this_cut.X.shape[0], train_this_cut.X.shape[1]))
        train_this_cut.X.to_csv(filename + "train_X_%d.csv" %
                                i, index=False, float_format="%.5f")
        train_this_cut.y.to_csv(filename + "train_y_%d.csv" %
                                i, index=False, float_format="%.5f")
        print("saving %svalid_X_%d.csv (%d row * %d col)"%(filename, i, valid_this_cut.X.shape[0], valid_this_cut.X.shape[1]))
        valid_this_cut.X.to_csv(filename + "valid_X_%d.csv" %
                                i, index=False, float_format="%.5f")
        valid_this_cut.y.to_csv(filename + "valid_y_%d.csv" %
                                i, index=False, float_format="%.5f")

if __name__ == "__main__":
    # read csv as dataframe
    filename = "./cv/cv_"
    print("loading data to memory...")
    train = pd.read_csv("./original/train.csv")
    test = pd.read_csv("./original/test.csv")
    # drop unrelated items
    train = dataset(train.drop("id", 1).drop("target", 1),
                    train.target.to_frame('target'))
    test = dataset(test.drop("id", 1), test.id.to_frame('id'))
    # -1 are actually NaN
    train.X = train.X.replace(-1, np.nan)
    test.X = test .X.replace(-1, np.nan)
    # do K-fold splitting
    KFOLDS = 5
    folds = list(StratifiedKFold(n_splits=KFOLDS, shuffle=True,
                                 random_state=10086).split(train.X, train.y))
    mydummylist, myheader = PreprocessOriginalData(train, test)
    KFoldsPreprocess(train, test, mydummylist, myheader)
