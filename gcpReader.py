import pandas as pd
import ast
from collections import Counter
import pandas as pd
import numpy as np
import time
import multiprocessing
from datetime import datetime
import ast
from collections import Counter
from numpy import loadtxt
from xgboost import XGBClassifier

from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import ast
from collections import Counter
import pandas as pd
import numpy as np
import time
import multiprocessing
from datetime import datetime
import ast
from collections import Counter
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class labelConverter():
    def __init__(self, df):
        self.df = df
        a = list(df.pt.unique())
        self.labelDict = {i: idx for idx, i in enumerate(a)}


    def convert_label(self, idx):
        currRow = self.df.irow[idx]
        currRow["pt"][idx] = self.labelDict[currRow["pt"][idx]]
        return currRow

    def conv_label_index(self,x):
        return self.labelDict[x]


    def run(self):

        # self.df = self.df.reset_index(drop=True)
        # indexes = list(range(len(self.df)))
        # pool = multiprocessing.Pool(processes=4)
        # result = pool.map(self.convert_label, indexes)
        # df = pd.concat([result], ignore_index=True)
        # pool.close()
        # pool.join()
        self.df["label"]=self.df.pt.apply(lambda x: self.conv_label_index(x))
        return self.df


class onehotReader():
    def __init__(self, aiDictPath="gs://abbvie_data_one/aggregate_full.csv", mode=False):
        rough_df = pd.read_csv(aiDictPath).dropna().reset_index(drop=True)
        rough_df = rough_df[rough_df.primaryid != "primaryid"]
        r = []
        [r.extend(ast.literal_eval(ai)) for ai in rough_df.pt]
        r = dict(Counter(r))
        self.reactionDict = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True) if v > 1000}

        ingredients = []
        [ingredients.extend(ast.literal_eval(ai)) for ai in rough_df.prod_ai]
        ingredients = dict(Counter(ingredients))
        self.aiDict = {k: v for k, v in sorted(ingredients.items(), key=lambda item: item[1], reverse=True) if v > 10}
        a = ['age', 'pt', 'sex', 'wt']
        columns = list(self.aiDict.keys())
        columns.extend(a)
        self.columns = columns
        self.mode = mode

    def readReaction(self, reaction):
        print("reading", reaction)
        df = pd.read_csv("gs://abbvie_data_one/{}OneHotEncoded.csv".format(reaction), names=self.columns, header=None)
        return df

    def read(self):
        if self.mode:
            reactions = list(self.reactionDict.keys())[:self.mode]
        else:
            reactions = list(self.reactionDict.keys())
        ###MULTIPROCESSING
        # print("multiprocessing starting")
        # pool = multiprocessing.Pool(processes=4)
        # result = pool.map(self.readReaction, reactions)
        # print("all df read")
        # df = pd.concat([result], ignore_index=True)
        # pool.close()
        # pool.join()

        ####Non multithreading
        df=pd.concat([pd.read_csv("gs://abbvie_data_one/{}OneHotEncoded.csv".format(reaction),
                               names=self.columns, header=None)\
                   for reaction in reactions], ignore_index=True)
        ##Convert labels
        finalDF = labelConverter(df).run()

        return finalDF





