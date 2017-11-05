# do preprocessing on raw data
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class dataset:
    def __init__(self, data=None, label=None):
        self.X = data
        self.y = label


class Preprocessing(object):
    # in order to let all dummies have same number of catagories
    @staticmethod
    def makehead(df, mydummylist=None):
        myheader = pd.DataFrame.copy(df.head(10))
        if mydummylist is None:
            mydummylist = self.dummylist(pd.concat(df, axis=0))
        assert(isinstance(df, pd.core.frame.DataFrame))
        assert(isinstance(mydummylist, list))

        for columnname in mydummylist:
            possible_val = list(df[columnname].unique())
            assert(len(possible_val) < 10)
            for i, val in enumerate(possible_val):
                myheader[columnname][i] = val
        return myheader

    @staticmethod
    def addhead(df, myheader):
        return pd.concat([myheader, df], axis=0, ignore_index=True)

    @staticmethod
    def rmhead(df):
        # drop the header rows, ret_index to start from 0, and drop previous indexs(with offset)
        return df.drop(range(10)).reset_index(drop=True)

    @staticmethod
    def fillNA(df, param=None):
        if param is None:
            df = df.fillna(-1)
            return df, param

    @staticmethod
    def dummylist(df):
        assert(isinstance(df, pd.core.frame.DataFrame))
        dummylist = []
        for columnname in list(df.columns):
            if columnname.endswith("cat") and len(list(df[columnname].unique())) < 10:
                dummylist.append(columnname)
        return dummylist

    @staticmethod
    def dummy(df, dummylist):
        assert(isinstance(df, pd.core.frame.DataFrame))
        for columnname in dummylist:
            newcolumn = pd.get_dummies(
                df[columnname], prefix="1hot_" + columnname[:-4])
            df = pd.concat([df, newcolumn], axis=1)
        for columnname in dummylist:
            df = df.drop(columnname, axis=1)
        return df

    @staticmethod
    def standardize(df, param=None):
        # unfinished
        return df, param


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
    # preprocessing training/testing
    print("preprocessing on whole training/testing set")
    train.X, NAmethod = Preprocessing.fillNA(train.X)
    test.X, _ = Preprocessing.fillNA(test.X, NAmethod)
    if True:
        # set down which list need 2b dummied
        mydummylist = Preprocessing.dummylist(
            pd.concat([train.X, test.X], axis=0))
        myheader = Preprocessing.makehead(
            pd.concat([train.X, test.X], axis=0), mydummylist)
    train.X = Preprocessing.dummy(
        Preprocessing.addhead(train.X, myheader), mydummylist)
    train.X = Preprocessing.rmhead(train.X)
    test.X = Preprocessing.dummy(
        Preprocessing.addhead(test.X, myheader), mydummylist)
    test.X = Preprocessing.rmhead(test.X)
    train.X, STDmethod = Preprocessing.standardize(train.X)
    test.X, _ = Preprocessing.standardize(test.X, STDmethod)
    train.X.to_csv(filename + "train_X.csv",
                   index=False, float_format="%.5f")
    train.y.to_csv(filename + "train_y.csv",
                   index=False, float_format="%.5f")
    test.X.to_csv(filename + "test_X.csv",
                  index=False, float_format="%.5f")
    test.y.to_csv(filename + "test_y.csv",
                  index=False, float_format="%.5f")
    # do methods in preprocessing in each fold
    for i, (train_idx, valid_idx) in enumerate(folds):
        print("preprocessing on cv #%d" % i)
        train_this_cut = dataset(
            train.X.loc[train_idx], train.y.loc[train_idx])
        valid_this_cut = dataset(
            train.X.loc[valid_idx], train.y.loc[valid_idx])

        train_this_cut.X, NAmethod = Preprocessing.fillNA(train_this_cut.X)
        train_this_cut.X = Preprocessing.dummy(
            Preprocessing.addhead(train_this_cut.X, myheader), mydummylist)
        train_this_cut.X = Preprocessing.rmhead(train_this_cut.X)
        train_this_cut.X, STDmethod = Preprocessing.standardize(
            train_this_cut.X)

        valid_this_cut.X, _ = Preprocessing.fillNA(
            valid_this_cut.X, NAmethod)
        valid_this_cut.X = Preprocessing.dummy(
            Preprocessing.addhead(valid_this_cut.X, myheader), mydummylist)
        valid_this_cut.X = Preprocessing.rmhead(valid_this_cut.X)
        valid_this_cut.X, _ = Preprocessing.standardize(
            valid_this_cut.X, STDmethod)

        train_this_cut.X.to_csv(filename + "train_X_%d.csv" %
                                i, index=False, float_format="%.5f")
        train_this_cut.y.to_csv(filename + "train_y_%d.csv" %
                                i, index=False, float_format="%.5f")
        valid_this_cut.X.to_csv(filename + "valid_X_%d.csv" %
                                i, index=False, float_format="%.5f")
        valid_this_cut.y.to_csv(filename + "valid_y_%d.csv" %
                                i, index=False, float_format="%.5f")
