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
    def RMOutliers(df):
        # df.y = df.y[df.X['ps_car_12']<1.2]
        df.y = df.y[df.X['ps_car_12']>-.5]
        # df.X = df.X[df.X['ps_car_12']<1.2]
        df.X = df.X[df.X['ps_car_12']>-.5]
        df.y['target'] = df.y['target'].astype(np.int64)
        return df

    @staticmethod
    def fillNA(df, param={'method': 'groupmean-1'}):
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
        elif param['method'] == 'groupmean-1':
            #fill catagory and binary with -1
            #fill other with mean(after grouping using the highest ranked catagory)
            group_criteria = ["ps_ind_05_cat",  #3rd     3classes
                                # "ps_car_01_cat",#9th    13classes       
                                # "ps_car_07_cat",#10th    3classes
                                "ps_ind_17_bin",#11th    2classes
                                # "ps_car_03_cat",#12th    3classes
                                ]
            for col in df.columns:
                if col.endswith('cat') or col.endswith('bin'):
                    df[col] = df[col].fillna(-1)
            if 'mean' not in param:
                param['mean_bak'] = df.mean()
                param['mean'] = df.groupby(group_criteria).mean()    
            new_df = []
            for possible_val, group in df.groupby(group_criteria):
                try:#possible that catagory not appeared
                    new_df.append(group.fillna(param['mean'].loc[possible_val]))
                except:
                    pass
            df = pd.concat(new_df, axis = 0).sort_index()
            #in case if there's empty catagory
            df = df.fillna(param['mean_bak'])
            return df, param
            
        else:
            raise KeyError("wrong method provided")

    @staticmethod
    def encoding(Dset, encoding_lst, keep_original = False,
                param = {'min_samples_leaf' : 200,
                        'smoothing' : 10,
                        'noise_level' : 1e-2}):
        """
        implementation of Empirical Bayesian Encoding
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        modified from https://www.kaggle.com/aharless/xgboost-cv-lb-284
        """
        def add_noise(series, noise_level):
            return series * (1 + noise_level * np.random.randn(len(series)))

        if 'average' not in param or 'prior' not in param:
            target = Dset.y
            if isinstance(target, pd.core.frame.DataFrame):
                target = target['target']
            min_samples_leaf = param['min_samples_leaf'] 
            smoothing = param['smoothing']
            # Apply average function to all target data
            param['target_name'] = target.name
            param['prior'] = target.mean()
            param['average'] = {}
            for col in encoding_lst:
                selected_series = Dset.X[col]
                assert len(selected_series) == len(target)
                temp = pd.concat([selected_series, target], axis=1)
                # Compute target mean
                averages = temp.groupby(by=selected_series.name)[param['target_name']].agg(["mean", "count"])
                # Compute smoothing
                smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
                # The bigger the count the less full_avg is taken into account
                averages[param['target_name']] = param['prior'] * (1 - smoothing) + averages["mean"] * smoothing
                averages.drop(["mean", "count"], axis=1, inplace=True)
                param['average'][col] = averages
        else:
            param['noise_level'] = 0

        for col in encoding_lst:
            selected_series = Dset.X[col]
            # Apply averages to series
            ft_selected_series = pd.merge(
                selected_series.to_frame(selected_series.name),
                param['average'][col].reset_index().rename(columns={'index': param['target_name'], param['target_name']: 'average'}),
                on=selected_series.name,
                how='left')['average'].rename(selected_series.name + '_mean').fillna(param['prior'])
            ft_selected_series.index = selected_series.index
            Dset.X[col+'_avg'] = add_noise(ft_selected_series, param['noise_level'])
            if not keep_original:
                Dset.X.drop([col], axis=1, inplace=True)
        return Dset.X, param

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
