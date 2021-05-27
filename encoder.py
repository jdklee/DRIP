import pandas as pd
import numpy as np
import time
import multiprocessing
from datetime import datetime

import ast
from collections import Counter
import copy
#from category_encoders import TargetEncoder

import os
from io import StringIO
from google.cloud import storage
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True )


class oneHotEncoder():
    def __init__(self, df, aiThreshold, labelThreshold, savePath, mode=False,
                 hotcode=False, reactioncode=False, both=False):
        print("No drop encoder")
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.aiThreshold = aiThreshold
        self.labelThreshold = labelThreshold
        self.startTime = time.time()
        self.savePath = savePath
        self.hotcode = hotcode
        self.reactioncode = reactioncode
        self.both = both
        if both:
            self.features = df.drop("pt", axis=1).reset_index(drop=True)
            self.label = df[["primaryid","caseid","pt"]].reset_index(drop=True)
            reactcol=["primaryid","caseid"]

            reactcol.extend(list(reactioncode))
            self.processedLabel=pd.DataFrame(columns=reactcol)

            columns=["primaryid","caseid","sex", "age", "wt"]
            columns.extend(list(self.hotcode))
            self.processedFeatures=pd.DataFrame(columns=columns)


            self.processedFeatures.to_csv("Features_"+self.savePath , index=False, header=True)
            self.processedLabel.to_csv("Labels_"+self.savePath,index=False, header=True)
        else:
            a = ['primaryid','caseid_x','age', 'sex', 'wt', 'pt']
            a.extend(self.hotcode)
            columns = a
            self.ret = pd.DataFrame(columns=columns)



    def setupLabels(self):
        print("setting up labels")
        r=[]
        [r.extend(ast.literal_eval(ai)) for ai in self.df.pt]
        r=dict(Counter(r))
        reactionList=list(set([k for (k,v) in r.items() if v>self.labelThreshold]))
        self.reactionList=reactionList

    # def setupColumnsAI(self,ai):
    #     self.df[ai]=0

    def setupColumns(self):
        print("setting up columns")
        startTime = time.time()
        if self.both:
            ais=self.hotcode
            for ai in ais:
                self.features[ai]=[0]*len(self.features)
            reactions = self.reactioncode
            for reaction in reactions:
                self.label[reaction] = [0] * len(self.label)

        else:
            if not self.hotcode:
                c = []

                [c.extend(ast.literal_eval(ai)) for ai in self.df.prod_ai]
                c = dict(Counter(c))
                ais = list(set([k for (k, v) in c.items() if v > self.aiThreshold]))
            if self.hotcode:
                ais = self.hotcode
            for ai in ais:
                self.df[ai] = 0

        print("setup column finished in {} seconds".format(time.time() - startTime))

    def encodeIndex(self, index):
        startTime = time.time()
        if index % 10000 == 0:
            print("index encoding:",index)

        ais = ast.literal_eval(self.df.prod_ai[index])


        currRow = self.df.iloc[index]
        if self.both:
            reactions = ast.literal_eval(self.df.pt[index])
            currRowFeatures = self.features.iloc[index]
            currRowLabels = self.label.iloc[index]
            #reactions = self.reactioncode.keys()

            labels={"primaryid":currRowLabels.primaryid,
                    "caseid":currRowLabels.caseid}
            features={"primaryid":currRowFeatures.primaryid,
                      "caseid":currRowFeatures.caseid,
                      "age":currRowFeatures.age,
                      "wt":currRowFeatures.wt,
                      "sex":currRowFeatures.sex}
            for ai in ais:
                #if ai in self.features.columns:
                features[ai] = 1
            for reaction in reactions:
                #if reaction in self.label.columns:
                labels[reaction] = 1
            #print(currRowFeatures)
            #print(currRowLabels)
            #currRowFeatures = currRowFeatures.drop(["caseid_x","prod_ai"])
            #currRowLabels = currRowLabels.drop(["caseid_x","pt"])
            #print(features)
            #print(labels)
            featdf=self.processedFeatures.append(pd.Series(currRowFeatures), ignore_index=True).fillna(0)
            featdf.to_csv("Features_"+self.savePath, mode="a", index=False, header=False)
            #print(featdf)
            labeldf=self.processedLabel.append(pd.Series(currRowLabels), ignore_index=True).fillna(0)
            labeldf.to_csv("Labels_"+self.savePath, mode="a",index=False, header=False)
            #print(labeldf)
            #return
        else:
            if not self.mode:
                reactions = ast.literal_eval(self.df.pt[index])
                for reaction in reactions:

                    temp = copy.deepcopy(currRow)
                    if reaction in self.reactionList:
                        temp.pt = reaction
                        for ai in ais:
                            if ai in self.df.columns:
                                temp[ai] = 1


                    temp = temp.drop(["prod_ai"])
                    self.ret.append(temp, ignore_index=True).to_csv(self.savePath, mode="a",
                                                                    index=False, header=False)
            else:

                temp = copy.deepcopy(currRow)
                # retDict={"primaryid": temp.primaryid,
                #  "caseid": temp.caseid,
                #  "age": temp.age,
                #  "wt": temp.wt,
                #  "sex": temp.sex,
                # "pt": temp.pt}
                for col in self.ret.columns:
                    if col in ais:
                        temp[col] = 1
                # for ai in ais:
                #     if ai in self.ret.columns:
                #         temp[ai] = 1

                temp = temp.drop(["prod_ai", "caseid_x"])
                self.ret.append(temp, ignore_index=True).to_csv(self.savePath, mode="a",
                                                                index=False, header=False)

    def encode(self):
        print("encoding active ingredients")
        self.df.reset_index(drop=True, inplace=True)
        indexes=list(range(len(self.df)))
        self.ret.to_csv(self.savePath, index=False, header=True)
        pool=multiprocessing.Pool(processes=9)
        pool.map(self.encodeIndex,indexes)
        pool.close()
        pool.join()

        # for i in indexes:
        #     self.encodeIndex(i)

        self.df.drop("prod_ai",axis=1,inplace=True)
        if not self.both:
            print('Time taken to encode AIs = {} seconds'.format(time.time() - self.startTime))
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/jdklee/Downloads/patentcitation-291203-3875e284f934.json'

            gcs = storage.Client()
            gcs.get_bucket('abbvie_data_one').blob("{}OneHotEncodedXGB.csv".format(self.mode)).upload_from_filename("/Users/jdklee/Documents/AbbVie/{}OneHotEncoded.csv".format(self.mode),
                                                                                     content_type='text/csv')
            print("save to gcloud successful {}".format(self.mode))

            os.remove("/Users/jdklee/Documents/AbbVie/{}OneHotEncoded.csv".format(self.mode))
            os.remove("/Users/jdklee/Documents/AbbVie/aggregate_{}.csv".format(self.mode))
        if self.both:
            print('Time taken to encode AIs = {} seconds'.format(time.time() - self.startTime))
            os.environ[
                'GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/jdklee/Downloads/patentcitation-291203-3875e284f934.json'

            gcs = storage.Client()
            gcs.get_bucket('abbvie_data_one').blob("oneHotEncodedLabels.csv").upload_from_filename\
                ("/Users/jdklee/Documents/AbbVie/oneHotEncodedLabels.csv".format(self.mode),
                                                                                     content_type='text/csv')
            gcs.get_bucket('abbvie_data_one').blob("oneHotEncodedFeatures.csv").upload_from_filename\
                ("/Users/jdklee/Documents/AbbVie/oneHotEncodedFeatures.csv".format(self.mode),
                                                                                     content_type='text/csv')
            print("save to gcloud successful {}".format(self.mode))
            os.remove("/Users/jdklee/Documents/AbbVie/{}OneHotEncoded.csv".format(self.mode))
            os.remove("/Users/jdklee/Documents/AbbVie/aggregate_{}.csv".format(self.mode))
    def run(self):
        if not self.both:
            self.setupColumns()
        if not self.mode and not self.both:
            self.setupLabels()
        self.encode()


"""
import pandas as pd
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['Animal Encoded'] = encoder.fit_transform(df['Animal'], df['Target'])
"""


class meanEncoder(oneHotEncoder):
    def __init__(self, df, aiThreshold, labelThreshold, savePath, mode=False):
        print("init mean encoder")
        oneHotEncoder.__init__(self, df, aiThreshold, labelThreshold, savePath, mode=mode)
        print("init done")
    def encode(self):
        print("encoding active ingredients")
        if "pair" in self.mode:
            catCol=["prod_ai1","prod_ai2"]
        if "single" in self.mode:
            catCol=["prod_ai"]
        self.calc_smooth_mean(df=self.df, catCol=catCol, target="pt", weight=300)
        print(self.smooth1)
        print(self.smooth2)
        indexes=list(range(len(self.df)))
        pool=multiprocessing.Pool(processes=8)
        pool.map(self.encodeIndex,indexes)
        pool.close()
        pool.join()

        print('Time taken to encode AIs = {} seconds'.format(time.time() - self.startTime))

    def encodeIndex(self, index):
        df = self.df
        smooth = self.smooth
        currRow = df.iloc[index]
        if "pair" in self.mode:
            currRow.prod_ai1[index] *= self.smooth1
            currRow.prod_ai2[index] *= self.smooth2
        if "single" in self.mode:
            currRow.prod_ai[index] *= self.smooth1
        if index<10:
            pd.DataFrame().append(currRow,ignore_index=True).to_csv(self.savePath, mode="a",
                                                                          index=False, header=True)
        else:
            pd.DataFrame().append(currRow, ignore_index=True).to_csv(self.savePath, mode="a",
                                                                     index=False, header=False)
        print("\r",index,"done")

    def calc_smooth_mean(self, df, catCol, target, weight):
        if "pair" in self.mode:
            mean = df[target].mean()
            agg = df.groupby(catCol[0])[target].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']

            # Compute the "smoothed" means
            self.smooth1 = (counts * means + weight * mean) / (counts + weight)

            mean = df[target].mean()
            agg = df.groupby(catCol[1])[target].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']

            self.smooth2 = (counts * means + weight * mean) / (counts + weight)
        if "single" in self.mode:
            mean = df[target].mean()
            agg = df.groupby(catCol[0])[target].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']

            # Compute the "smoothed" means
            self.smooth1 = (counts * means + weight * mean) / (counts + weight)

    def setupLabel(self):
        """given list of reactions, convert all labels to ints (duplicate rows)"""
        df = self.df
        r = list(df.pt)
        r = dict(Counter(r))
        reactionList = list(set([k for (k, v) in r.items() if v > self.labelThreshold]))
        self.reactionList = reactionList
        self.reactionDict = {}
        for idx, reac in enumerate(reactionList):
            self.reactionDict[reac] = idx

        self.df = df[df.pt.isin(reactionList)].reset_index(drop=True)

    def setupAIs(self):
        df = self.df
        if "pair" in self.mode:
            r = list(df.prod_ai1)
            r = dict(Counter(r))
            aiList = list(set([k for (k, v) in r.items() if v > self.aiThreshold]))
            self.df = df[df.prod_ai1.isin(aiList)].reset_index(drop=True)
        else:
            r = list(df.prod_ai)
            r = dict(Counter(r))
            aiList = list(set([k for (k, v) in r.items() if v > self.aiThreshold]))
            self.df = df[df.prod_ai1.isin(aiList)].reset_index(drop=True)

    def integerEncodeLabel(self):
        df = self.df

        for idx, reac in df.pt:
            intRep = self.reactionDict[reac]
            df.pt[idx] = intRep
        self.df = df

    def run(self):
        self.setupAIs()
        self.setupLabel()
        self.integerEncodeLabel()
        self.encode()
