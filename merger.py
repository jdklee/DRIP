import pandas as pd

import numpy as np
import time
import multiprocessing
class AImerger():
    def __init__(self,drug,react):
        self.drug=drug
        self.react=react

    def mergeAIsimple(self):
        
        startTime=time.time()
        df=self.drug.drop_duplicates()
        df=df.groupby(["primaryid"])['prod_ai'].apply(list)
        self.pairDrug=df
        #print("merge ai df=\n",df)
        
        print('Time taken to merge AIs = {} seconds'.format(time.time() - startTime))
        return df
    

    
    def mergeReactionSimple(self):
        
        startTime=time.time()
        df=self.react.drop_duplicates()

        df=df.groupby(["primaryid"])['pt'].apply(list)
        print('Time taken to merge reactions = {} seconds'.format(time.time() - startTime))
        self.pairReact=df
        return df
        
 
    
    
class dataSetup():
    
    
    def __init__(self, demo, drug, reac):
        drug=drug[drug.primaryid!="\n"]
        drug["primaryid"]=drug.primaryid.apply(lambda x: int(float(x)))
        drug["caseid"]=drug.caseid.apply(lambda x: int(float(x)))
        drug.dropna(inplace=True)
        demo=demo[demo.primaryid!="\n"]
        demo["primaryid"]=demo.primaryid.apply(lambda x: int(float(x)))

        demo["caseid"]=demo.caseid.apply(lambda x: int(float(x)))
        demo.dropna(inplace=True)

        demo['sex']=demo.sex.apply(lambda x: 0 if x=="M" else 1)

        reac=reac[reac.primaryid!="\n"]
        reac["primaryid"]=reac.primaryid.apply(lambda x:int(float(x)))
        reac["caseid"]=reac.caseid.apply(lambda x: int(float(x)))

        reac.dropna(inplace=True)
        self.demo=demo.reset_index(drop=True)
        self.setupAge()
        self.drug=drug.reset_index(drop=True)
        self.reac=reac.reset_index(drop=True)
    def setupAge(self):
        for i,n in enumerate(self.demo['age']):
            if n<17:
                self.demo['age'][i]=1
            elif n>=17 and n<=24:
                self.demo['age'][i]=2
            elif n>=24 and n<=65:
                self.demo['age'][i]=3
            else:
                self.demo['age'][i]=4
        
    def convert_kg_to_lb(self,df):
        for i in range(len(df)):
            if df.wt_cod[i]=="LBS":
                df.wt[i]=0.453592*float(list(df.wt)[i])

    def mergeTables(self):
        drug_filtered=self.drug
        demo_filtered=self.demo
        if "wt_cod" in demo_filtered.columns:
            self.convert_kg_to_lb(demo_filtered)
            demo_filtered.drop("wt_cod", axis=1, inplace=True)
            
        reaction=self.reac
        a=AImerger(drug=drug_filtered,react=reaction)
        pairDrug=a.mergeAIsimple()
        pairReact=a.mergeReactionSimple()

        self.pairDrug=pairDrug
        self.pairReact=pairReact
        #print("pairdrug:\n",pairDrug)
        #print("pairReaact:\n",pairReact)
              
        first_df=demo_filtered.merge(pairDrug.dropna(), how="inner", on=["primaryid"])
        roughDF=first_df.merge(pairReact.dropna(),how="inner", on=["primaryid"])
        
        self.finalDF=roughDF
        
        
        return roughDF
    


        
    def clean(self):
        return self.mergeTables()
    #setupDataMeanEncoding(demo, drug, reac, reactionType, mode, labelThreshold)

class setupDataMeanEncoding(dataSetup):
    def __init__(self, demo, drug, reac, mode,):
        print("init starting")
        dataSetup.__init__(self,  demo=demo, drug=drug, reac=reac)
        print("dataSetup init done")
        self.mode=mode
    def pairwiseDrugMerge(self,drug):
        drug=drug.drop_duplicates().reset_index(drop=True)
        pids=list(drug.primaryid)
        cids=list(drug.caseid)
        pidDF=[]
        cidDF=[]
        ai1=[]
        ai2=[]
        startTime=time.time()
        for index,(pid,cid) in enumerate(zip(pids,cids)):
            pidMemo=set()
            if index%1000==0:
                print(index,"out of",len(drug),"completed")
            temp=drug[drug.primaryid==pid].drop_duplicates().reset_index(drop=True)
            for i in range(len(temp)):
                for j in range(i+1,len(temp)):
                    if (temp.prod_ai[i],temp.prod_ai[j]) in pidMemo:
                        continue
                    if temp.prod_ai[i]==temp.prod_ai[j]:
                            continue
                    else:   
                        pidDF.append(pid)
                        cidDF.append(cid)
                        ai1.append(temp.prod_ai[i])
                        ai2.append(temp.prod_ai[j])
                        pidMemo.add((temp.prod_ai[i],temp.prod_ai[j]))
                        pidMemo.add((temp.prod_ai[j],temp.prod_ai[i]))
        ret=pd.DataFrame(columns=["primaryid","caseid","prod_ai1","prod_ai2"])
        ret["primaryid"]=pidDF
        ret["caseid"]=cidDF
        ret["prod_ai1"]=ai1
        ret["prod_ai2"]=ai2
        ret=ret.drop_duplicates().reset_index(drop=True)
        print('Time taken = {} seconds'.format(time.time() - startTime))
        return ret    
    
    def mergeTables(self):
        if "pair" in self.mode:
            print("pair merge in prog")
            pairDrug=self.pairwiseDrugMerge(self.drug)
            print("merge drug pair done")
            first_df=self.demo.merge(pairDrug, how="left", on=["primaryid","caseid"])
            finalDF=first_df.merge(self.reaction,how="left", on=["primaryid","caseid"]).drop_duplicates()
            return finalDF
        if "single" in self.mode:
            first_df=self.demo.merge(self.drug, how="left", on=["primaryid","caseid"])
            finalDF=first_df.merge(self.reaction, how="left", on=["primaryid","caseid"]).drop_duplicates()
            return finalDF
     

                        
            
        
        
        
class setupDataFilter(dataSetup):
    def __init__(self, demo, drug, reac, reactionType):
        print("init starting")
        dataSetup.__init__(self,  demo=demo, drug=drug, reac=reac)
        print("superclass made")
        self.reactionType=reactionType
        self.reac=self.reac[self.reac.pt==self.reactionType]
        print("reac filtered")
        print(self.reac)
        print("init successful for DataFilter child class")
    def mergeTables(self):
        drug_filtered=self.drug
        demo_filtered=self.demo
        reac_filtered=self.reac
        if "wt_cod" in demo_filtered.columns:
            self.convert_kg_to_lb(demo_filtered)
            demo_filtered.drop("wt_cod", axis=1, inplace=True)
            
        
        a=AImerger(drug=drug_filtered,react=reac_filtered)
        pairDrug=a.mergeAIsimple()
        self.pairDrug=pairDrug

              
        first_df=reac_filtered.merge(demo_filtered.dropna(), how="inner", on=["primaryid"])
        roughDF=first_df.merge(pairDrug.dropna(),how="inner", on=["primaryid"])
        
        self.finalDF=roughDF
        
        
        return roughDF
        
        
        