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
        myheader = pd.DataFrame.copy(df.head(200))
        if mydummylist is None:
            mydummylist = self.dummylist(pd.concat(df, axis=0))
        assert(isinstance(df, pd.core.frame.DataFrame))
        assert(isinstance(mydummylist, list))

        for columnname in mydummylist:
            possible_val = list(df[columnname].unique())
            assert(len(possible_val) < 200)
            for i, val in enumerate(possible_val):
                myheader[columnname][i] = val
        return myheader

    @staticmethod
    def addhead(df, myheader):
        return pd.concat([myheader, df], axis=0, ignore_index=True)

    @staticmethod
    def rmhead(df):
        # drop the header rows, ret_index to start from 0, and drop previous indexs(with offset)
        return df.drop(range(200)).reset_index(drop=True)

    @staticmethod
    def fillNA(df, param={'method': 'mean-1'}):
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
        elif param['method'] == 'mean-1':
            #fill catagory and binary with -1
            #fill other with mean
            for col in df.columns:
                if col.endswith('cat') or col.endswith('bin'):
                    df[col] = df[col].fillna(-1)
            if 'mean' not in param:
                param['mean'] = df.mean()
            df = df.fillna(param['mean'])
            return df, param
        else:
            raise KeyError("wrong method provided")

    @staticmethod
    def dummylist(df, uplimit = 99999):
        assert(isinstance(df, pd.core.frame.DataFrame))
        dummylist = []
        for columnname in list(df.columns):
            if columnname.endswith("cat") and len(list(df[columnname].unique())) < uplimit:#10
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
    def standardize(df, param={'method':'minmax'}):
        if param['method'] == 'minmax':
            if not 'min' in param or not 'max' in param:
                param['min'] = {}
                param['max'] = {}
                for col in df.columns:
                    if not col.endswith('bin') and not col.startswith('1hot'):
                        param['min'][col] = df[col].min()
                        param['max'][col] = df[col].max()
            for col in df.columns:
                if not col.endswith('bin') and not col.startswith('1hot'):
                    df[col] = (df[col]-param['min'][col])/(param['max'][col]-param['min'][col])
            return df, param
        elif param['method'] == 'meanstd':
            if not 'mean' in param or not 'std' in param:
                param['mean'] = df.mean()
                param['std'] = df.std()
            return (df-param['mean'])/param['std'], param
        else:
            return df, param
