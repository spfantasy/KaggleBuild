# do preprocessing on raw data
import warnings 
warnings.filterwarnings("ignore")
import pandas as pd
import numpy  as np
from sklearn.model_selection import StratifiedKFold

class dataset:
    def __init__(self, data = None, label = None):
        self.X = data
        self.y = label

class Preprocessing(object):
    # in order to let all dummies have same number of catagories
    @staticmethod
    def makehead(df):
        pass
    @staticmethod
    def addhead(df):
        pass
    @staticmethod
    def rmhead(df):
        pass
    @staticmethod
    def fillNA(df, param = None):
        if param is None:
            df = df.fillna(-1)
            return df, param
    @staticmethod
    def dummylist(df):
        assert(isinstance(df,pd.core.frame.DataFrame))
        dummylist = []
        for columnname in list(df.columns):
            if columnname.endswith("cat") and len(list(df[columnname].unique())) < 10:
                dummylist.append(columnname)
        return dummylist
    @staticmethod
    def dummy(df, dummylist):
        assert(isinstance(df,pd.core.frame.DataFrame))
        for columnname in dummylist:
            newcolumn = pd.get_dummies(df[columnname],prefix = "1hot_"+columnname[:-4])
            df = pd.concat([df, newcolumn],axis = 1)
        for columnname in dummylist:
            df = df.drop(columnname, axis = 1)
        return df
    @staticmethod
    def standardize(df, param = None):
        #unfinished
        return df, param

if __name__ == "__main__":
    # read csv as dataframe
    print("loading data to memory...")
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    # drop unrelated items
    train = dataset(train.drop("id",1).drop("target",1), train.target.to_frame('target'))
    test  = dataset(test.drop("id",1), test.id.to_frame('id'))
    # -1 are actually NaN
    train.X = train.X.replace(-1, np.nan)
    test.X  = test .X.replace(-1, np.nan)
    # do K-fold splitting
    KFOLDS = 5
    folds = list(StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=2016).split(train.X, train.y))
    mydummylist = Preprocessing.dummylist(train.X)
    # do methods in preprocessing in each fold   
    for i, (train_idx, valid_idx) in enumerate(folds):
        print("preprocessing on cv #%d"%i)
        filename = "./cv/cv_"
        train_this_cut = dataset(train.X.loc[train_idx], train.y.loc[train_idx])
        valid_this_cut = dataset(train.X.loc[valid_idx], train.y.loc[valid_idx])
        
        train_this_cut.X, NAmethod  = Preprocessing.fillNA(train_this_cut.X)
        train_this_cut.X            = Preprocessing.dummy(train_this_cut.X, mydummylist)
        train_this_cut.X, STDmethod = Preprocessing.standardize(train_this_cut.X)

        valid_this_cut.X, _ = Preprocessing.fillNA(valid_this_cut.X, NAmethod)
        valid_this_cut.X    = Preprocessing.dummy(valid_this_cut.X, mydummylist)
        valid_this_cut.X, _ = Preprocessing.standardize(valid_this_cut.X, STDmethod)

        train_this_cut.X.to_csv(filename+"train_X_%d.csv"%i, index = False, float_format = "%.5f")
        train_this_cut.y.to_csv(filename+"train_y_%d.csv"%i, index = False, float_format = "%.5f")
        valid_this_cut.X.to_csv(filename+"valid_X_%d.csv"%i, index = False, float_format = "%.5f")
        valid_this_cut.y.to_csv(filename+"valid_y_%d.csv"%i, index = False, float_format = "%.5f")
    # preprocessing testing
    print("preprocessing on whole training set")
    train.X, NAmethod  = Preprocessing.fillNA(train.X)
    train.X            = Preprocessing.dummy(train.X, mydummylist)
    train.X, STDmethod = Preprocessing.standardize(train.X)
    train.X.to_csv(filename+"train_X.csv", index = False, float_format = "%.5f")
    train.y.to_csv(filename+"train_y.csv", index = False, float_format = "%.5f")
    print("preprocessing on whole testing set")
    test.X, NAmethod  = Preprocessing.fillNA(test.X)
    test.X            = Preprocessing.dummy(test.X, mydummylist)
    test.X, STDmethod = Preprocessing.standardize(test.X)
    test.X.to_csv(filename+"test_X.csv", index = False, float_format = "%.5f")
    test.y.to_csv(filename+"test_y.csv", index = False, float_format = "%.5f")
