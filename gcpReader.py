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
    def __init__(self, df, binary=False, target=False):
        self.df = df
        self.binary=binary
        self.target=target
        a = list(df.pt.unique())
        self.labelDict = {i: idx for idx, i in enumerate(a)}



    def conv_label_index(self,x):
        if self.binary:
            if x == self.target:
                return 1
            else:
                return 0
        else:
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
    def __init__(self, aiDictPath="gs://abbvie_data_one/aggregate_full.csv", mode=False, binary=False, target=False):
        self.binary=binary
        self.restList=[]
        self.target=target
        self.restDF=pd.DataFrame()
        self.targetDF=pd.DataFrame()
        rough_df = pd.read_csv(aiDictPath).dropna().reset_index(drop=True)
        rough_df = rough_df[rough_df.primaryid != "primaryid"]
        r = []
        [r.extend(ast.literal_eval(ai)) for ai in rough_df.pt]
        r = dict(Counter(r))
        self.reactionDict = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True) if v > 10000}

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

    def random_pick(self,reactionList):
        import random
        temp = reactionList

        #remove all used reactions
        temp.remove(self.target)
        [temp.remove(i) for i in self.restList]
        random_index = random.randint(0, len(temp) - 1)
        self.restList.append(reactionList[random_index])
        return reactionList[random_index]

    def read(self):
        if self.binary:
            self.targetDF=self.readReaction(self.target)
            alert=False
            #alarm for when cond is met
            while not alert:
                rest = self.random_pick(list(self.reactionDict.keys()))
                try:
                    to_append = self.readReaction(rest)
                    print("length of target:",len(self.targetDF))
                    print("length of to append+ restDF:",len(to_append)+len(self.restDF))

                    if len(to_append)+len(self.restDF) < len(self.targetDF)+400:
                        print("appending to restDF")
                        self.restDF=self.restDF.append(to_append)
                        #print("length of others df after append:", len(self.restDF))

                    if 0.60 > len(self.targetDF)/(len(self.targetDF)+len(self.restDF)):
                        if len(self.targetDF) / (len(self.targetDF)+len(self.restDF)) > 0.40:
                            total_length=len(self.restDF)+len(self.targetDF)
                            print("found the golden ratio:", len(self.targetDF)/total_length)
                            alert = True
                except:
                    print(rest, " is not found in bucket")

            #Keep picking new reactions until ratio is at around 0.5
  #           label_ratio=len(self.targetDF)/(len(self.targetDF)+len(self.restDF))
  #           while label_ratio>0.7 or label_ratio<0.4:
  #               rest=self.random_pick(list(self.reactionDict.keys()))
  #               to_append=self.readReaction(rest)
  #               temp=self.restDF
  # ### Do not append to rest if that unbalances the dataset
  #               temp_label_ratio = len(self.targetDF)/(len(self.targetDF)+len(self.restDF) + len(temp))
  #               print("temporary label ratio after merging {}:".format(rest), temp_label_ratio)
  #               if temp_label_ratio<0.4:
  #                   continue
  #               else:
  #                   self.restDF = self.restDF.append(to_append)
  #                   label_ratio = temp_label_ratio
  #                   print("label ratio =",label_ratio)
  #                   print("lengh of restDF: ", len(self.restDF))

            #When ratio is right, concat and return
            df = pd.concat([self.targetDF,self.restDF], ignore_index=True)
            print("length of df after concat:", len(df))
            #df["Label"]=df.pt.apply(lambda x: 1 if x == self.target else 0)
            df = labelConverter(df, binary=True, target=self.target).run()
            print("length of df after converting labels:", len(df))
            return df

        else:
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





