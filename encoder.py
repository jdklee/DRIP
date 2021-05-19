import pandas as pd
import numpy as np
import time
import multiprocessing
from datetime import datetime

import ast
from collections import Counter
import copy
#from category_encoders import TargetEncoder


        
    
        
        
                
                
class oneHotEncoder():
    def __init__(self, df, aiThreshold, labelThreshold, savePath, mode=False):
        self.df=df.reset_index(drop=True)
        self.mode=mode
        self.aiThreshold=aiThreshold
        self.labelThreshold=labelThreshold
        self.startTime=time.time()
        self.savePath=savePath
        
        
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
        c=[]
        startTime=time.time()
        [c.extend(ast.literal_eval(ai)) for ai in self.df.prod_ai]
        c=dict(Counter(c))
        ais=list(set([k for (k,v) in c.items() if v>self.aiThreshold]))
        for ai in ais:
            self.df[ai]=0
        print(self.df.columns)
        print("setup column finished in {} seconds".format(time.time()-startTime))

    def encodeIndex(self,index):
        startTime=time.time()
        if index%1000==0:
            print(index)
        if index<10:

            header=True
        if index>=10:
            header=False
        
        ais=ast.literal_eval(self.df.prod_ai[index])

        currRow=self.df.iloc[index]
        if not self.mode:
            reactions = ast.literal_eval(self.df.pt[index])
            for reaction in reactions:

                temp=copy.deepcopy(currRow)
                if reaction in self.reactionList:
                    temp.pt=reaction
                    for ai in ais:
                        if ai in self.df.columns:
                            temp[ai]=1
                temp=temp.drop(["primaryid","caseid","prod_ai"])
                pd.DataFrame().append(temp, ignore_index=True).to_csv(self.savePath, mode="a",
                                                                      index=False, header=header)
        else:
            temp=copy.deepcopy(currRow)
            for ai in ais:
                if ai in self.df.columns:
                    temp[ai]=1
            temp = temp.drop(["primaryid", "caseid_x", "prod_ai"])
            pd.DataFrame().append(temp, ignore_index=True).to_csv(self.savePath, mode="a",
                                                                  index=False, header=header)

               
    
    def encode(self):
        print("encoding active ingredients")
        indexes=list(range(len(self.df)))
        pool=multiprocessing.Pool(processes=10)
        pool.map(self.encodeIndex,indexes)
        pool.close()
        pool.join()
        self.df.drop("prod_ai",axis=1,inplace=True)
        print('Time taken to encode AIs = {} seconds'.format(time.time() - self.startTime))

    

    def run(self):
        self.setupColumns()
        if not self.mode:
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
        oneHotEncoder.__init__(self, df, aiThreshold, labelThreshold, savePath, mode=mode)

    def encode(self):
        print("encoding active ingredients")
        if "pair" in self.mode:
            catCol=["prod_ai1","prod_ai2"]
        if "single" in self.mode:
            catCol=["prod_ai"]
        self.calc_smooth_mean(df=self.df, catCol=catCol, target="pt", weight=300)
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
