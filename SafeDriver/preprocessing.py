from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")
from modeling import dataset
import pandas as pd
import numpy as np


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
    def fillNA(df, param={'method': 'drop'}):
        if param['method'] == 'fill-1':
            df = df.fillna(-1)
            return df, param
        elif param['method'] == 'drop':
            if 'droplist' not in param:
                droplist = []
                for col in df.columns:
                    if df[col].hasnans:
                        droplist.append(col)
                param['droplist'] = droplist
            for col in param['droplist']:
                df = df.drop(col, axis=1)
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
