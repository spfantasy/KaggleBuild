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

def firststeps(df, param = None):
    def addcolumns(df):
        # df['ps_c13*ps_r03'] = df['ps_car_13'] * df['ps_reg_03']
        # df['missing_vals'] = np.sum((df==np.nan).values, axis=1)

        # df['ps_car_13_sq2'] = df['ps_car_13'] * df['ps_car_13']
        # df['ps_reg_03_sq2'] = df['ps_reg_03'] * df['ps_reg_03']
        return df
    def selectfeatures(df):
        #feature shadowing from Oliver
        train_features = [
            "ps_car_13",  #            : 1571.65 / shadow  609.23
            "ps_reg_03",  #            : 1408.42 / shadow  511.15
            "ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
            "ps_ind_03",  #            : 1219.47 / shadow  230.55
            "ps_ind_15",  #            :  922.18 / shadow  242.00
            "ps_reg_02",  #            :  920.65 / shadow  267.50
            "ps_car_14",  #            :  798.48 / shadow  549.58
            "ps_car_12",  #            :  731.93 / shadow  293.62
            "ps_car_01_cat",  #        :  698.07 / shadow  178.72
            "ps_car_07_cat",  #        :  694.53 / shadow   36.35
            "ps_ind_17_bin",  #        :  620.77 / shadow   23.15
            "ps_car_03_cat",  #        :  611.73 / shadow   50.67
            "ps_reg_01",  #            :  598.60 / shadow  178.57
            "ps_car_15",  #            :  593.35 / shadow  226.43
            "ps_ind_01",  #            :  547.32 / shadow  154.58
            "ps_ind_16_bin",  #        :  475.37 / shadow   34.17
            "ps_ind_07_bin",  #        :  435.28 / shadow   28.92
            "ps_car_06_cat",  #        :  398.02 / shadow  212.43
            "ps_car_04_cat",  #        :  376.87 / shadow   76.98
            "ps_ind_06_bin",  #        :  370.97 / shadow   36.13
            "ps_car_09_cat",  #        :  214.12 / shadow   81.38
            "ps_car_02_cat",  #        :  203.03 / shadow   26.67
            "ps_ind_02_cat",  #        :  189.47 / shadow   65.68
            "ps_car_11",  #            :  173.28 / shadow   76.45
            "ps_car_05_cat",  #        :  172.75 / shadow   62.92
            "ps_calc_09",  #           :  169.13 / shadow  129.72
            "ps_calc_05",  #           :  148.83 / shadow  120.68
            "ps_ind_08_bin",  #        :  140.73 / shadow   27.63
            "ps_car_08_cat",  #        :  120.87 / shadow   28.82
            "ps_ind_09_bin",  #        :  113.92 / shadow   27.05
            "ps_ind_04_cat",  #        :  107.27 / shadow   37.43
            "ps_ind_18_bin",  #        :   77.42 / shadow   25.97
            "ps_ind_12_bin",  #        :   39.67 / shadow   15.52
            "ps_ind_14",  #            :   37.37 / shadow   16.65
            "ps_car_11_cat",
        ]
        return df[train_features]
    df = selectfeatures(df)
    df = addcolumns(df)
    return df


def PreprocessOriginalData(train, test, filename):
    # preprocessing training/testing
    print("preprocessing on whole training/testing set")
    # deepcopy the train/test set to preprocess
    from copy import deepcopy
    train = deepcopy(train)
    test = deepcopy(test)
    test.X = pd.DataFrame.copy(test.X)
    # feature engineering
    train.X = firststeps(train.X)
    test.X = firststeps(test.X)
    # fillNA
    train.X, NAmethod = PP.fillNA(train.X)
    test.X, _ = PP.fillNA(test.X, NAmethod)
    # Empirical Bayesian Encoding
    encoding_lst = [col for col in train.X.columns if col.endswith('cat') 
                    and len(pd.concat([train.X, test.X])[col].unique()) >= 13]
    train.X, Encodingmethod = PP.encoding(train, encoding_lst)
    test.X, _ = PP.encoding(test, encoding_lst, param = Encodingmethod)
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
    return mydummylist, myheader, encoding_lst

def KFoldsPreprocess(train, test, mydummylist, myheader, encoding_lst, KFOLDS, filename):
    folds = list(StratifiedKFold(n_splits=KFOLDS, shuffle=True,
                             random_state=10086).split(train.X, train.y))
    # do methods in preprocessing in each fold
    for i, (train_idx, valid_idx) in enumerate(folds):
        print("preprocessing on cv #%d" % i)
        train_this_cut = dataset(
            train.X.iloc[train_idx], train.y.iloc[train_idx])
        valid_this_cut = dataset(
            train.X.iloc[valid_idx], train.y.iloc[valid_idx])
        # feature engineering
        train_this_cut.X = firststeps(train_this_cut.X)
        valid_this_cut.X = firststeps(valid_this_cut.X)
        # fillNA
        train_this_cut.X, NAmethod = PP.fillNA(train_this_cut.X)
        # Empirical Bayesian Encoding
        train_this_cut.X, Encodingmethod = PP.encoding(train_this_cut, encoding_lst)
        train_this_cut.X = PP.dummy(
            PP.addhead(train_this_cut.X, myheader), mydummylist)
        train_this_cut.X = PP.rmhead(train_this_cut.X)
        train_this_cut.X, STDmethod = PP.standardize(
            train_this_cut.X)
        valid_this_cut.X, _ = PP.fillNA(
            valid_this_cut.X, NAmethod)
        valid_this_cut.X, _ = PP.encoding(valid_this_cut, encoding_lst, param = Encodingmethod)
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
    # rm outliers
    # train = PP.RMOutliers(train)
    # do K-fold splitting
    KFOLDS = 5
    mydummylist, myheader, encoding_lst = PreprocessOriginalData(train, test, filename = filename)
    KFoldsPreprocess(train, test, mydummylist, myheader, encoding_lst, KFOLDS, filename = filename)
