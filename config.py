# -*- coding: utf-8 -*-
import os
import csv
import time
import datetime
import random
import json

import warnings
from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
warnings.filterwarnings("ignore")


class Config(object): 
      seed = 300
      epoch = 10000
      lr = 0.00005
      weight_decay = 10**-2
      batch_size = 32
      step = 300
      gamma = 0.3
      seqLen = 200
      rate = 0.8
      
      comm_size = 60
      
      nclass=2 
      nlayer=3
      bi_num_hidden = 120
      nums_comm = 5
      nums_feat = 15
      nums_bert = 768
      nembd_1 = 32 
      nembd_2 = 8
    
      data_rumEval = './data/phemeTextAndLabelsRumEval2017.json'
      data_rumEval_test = './data/test2017.json'
      data_rumEval_dev = './data/rumoureval-subtaskB-dev.json'

      pheme_train = r"C:\Users\stat\Model\GEAR\GEAR081717\DTCA_FiGNN\interact_fusions\data\pheme_data_temporal\pheme_train.json"
      pheme_dev = r"C:\Users\stat\Model\GEAR\GEAR081717\DTCA_FiGNN\interact_fusions\data\pheme_data_temporal\pheme_dev.json"
      pheme_test = r"C:\Users\stat\Model\GEAR\GEAR081717\DTCA_FiGNN\interact_fusions\data\pheme_data_temporal\pheme_test.json"













    
    


