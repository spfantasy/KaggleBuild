from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
import pandas as pd
import numpy as np

from modeling import dataset

class Evaluation(object):
    @staticmethod
    def gini(y, pred):
        fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
        g = 2 * metrics.auc(fpr, tpr) -1
        return g