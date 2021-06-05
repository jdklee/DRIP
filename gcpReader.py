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
from google.cloud import storage
import os
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
        self.reaction = self.list_blobs()

    def list_blobs(self):
        """Lists all the blobs in the bucket."""
        from google.cloud import storage
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/jdklee/gcp/patentcitation-291203-3875e284f934.json'
        bucket_name = "abbvie_data_one"

        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(bucket_name)

        files = [blob.name for blob in blobs]
        files = [i.split("One")[0] for i in files if "aggregate" not in i]
        self.reactions=files
        #print(files)
        return files
    def readReaction(self, reaction, chunk=False):
        print("reading", reaction)
        try:
            if chunk:
                df=pd.read_csv("gs://abbvie_data_one/{}OneHotEncoded.csv".format(reaction),
                               names=self.columns,
                               nrows=chunk, header=None)
                return df
            else:
                df = pd.read_csv("gs://abbvie_data_one/{}OneHotEncoded.csv".format(reaction), names=self.columns, header=None)
                self.terminate=False
                return df
        except:
            print("One hot encoded don't exist")
            self.terminate=True

    def random_pick(self,reactionList):
        import copy
        import random
        temp = copy.deepcopy(reactionList)
        #remove all used reactions
        temp.remove(self.target)
        [temp.remove(i) for i in self.restList if i in temp]
        random_index = random.randint(0, len(temp) - 1)

        #edit distance to get most unrelated ones
        import nltk
        while nltk.edit_distance(temp[random_index], self.target) <10:
            random_index = random.randint(0, len(temp) - 1)
        return temp[random_index]

    def remove_duplicates(self,df):
        features=df.drop(["label","pt"],axis=1)
        to_drop=features[features.duplicated(keep="first")==True].index
        df.drop(list(to_drop), axis=0, inplace=True)
        print(len(to_drop),"rows dropped")
        return df

    def read(self):
        if self.binary:
            self.targetDF=self.readReaction(self.target)
            if 200 in self.targetDF.columns:
                print("target DF contains different columns")
                return False
            if self.terminate:
                return False
            alert=False
            #alarm for when cond is met
            while not alert:
                rest = self.random_pick(self.reaction)
                try:
                    to_append = self.readReaction(rest)
                    print("length of target:",len(self.targetDF))
                    print("length of to append+ restDF:",len(to_append)+len(self.restDF))
                    if len(self.targetDF)<10000 and len(to_append)>len(self.targetDF):
                        to_append = self.readReaction(rest, chunk=(len(self.targetDF)-len(self.restDF)))
                        print("small targetDF: ", len(self.targetDF))
                        print("appending to restDF")
                        self.restDF = self.restDF.append(to_append)
                        self.restList.append(rest)
                        print("append {} to restList".format(rest))
                            # print("length of others df after append:", len(self.restDF))
                        if 0.70 > len(self.targetDF)/(len(self.targetDF)+len(self.restDF)):
                            if len(self.targetDF) / (len(self.targetDF)+len(self.restDF)) > 0.30:
                                total_length=len(self.restDF)+len(self.targetDF)
                                print("found the golden ratio:", len(self.targetDF)/total_length)
                                alert = True
                                print("reset restList")
                                self.restList=[]
                    else:
                        if len(to_append)+len(self.restDF) < len(self.targetDF)+5000:
                            print("appending to restDF")
                            self.restDF=self.restDF.append(to_append)
                            self.restList.append(rest)
                            print("append {} to restList".format(rest))
                            #print("length of others df after append:", len(self.restDF))

                        if 0.60 > len(self.targetDF)/(len(self.targetDF)+len(self.restDF)):
                            if len(self.targetDF) / (len(self.targetDF)+len(self.restDF)) > 0.40:
                                total_length=len(self.restDF)+len(self.targetDF)
                                print("found the golden ratio:", len(self.targetDF)/total_length)
                                alert = True
                                print("reset restList")
                                self.restList=[]

                except Exception as e:
                    print(e)
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





