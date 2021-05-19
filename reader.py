
        
import pandas as pd
from merger import *
#from encoder import *
import numpy as np
import time
import multiprocessing
class dataReader():
    def __init__(self, chunkSize, drugs, reactions, demographics,mode=False):
        self.chunkSize=chunkSize
        
        self.mode=mode
        self.drugs=drugs
        self.reactions=reactions
        self.demographics=demographics
        self.first_one = True
        
    def reader(self):
        path=self.path
        fullpath=[self.path+folder+"/ascii/"+csvfile for folder in folders for csvfile in[i for i in os.listdir(path+folders[0]+"/ascii") if ".txt" in i and ("reac" in i.lower() or "drug" in i.lower() or "demo" in i.lower())]]
        
#    def appender(self, file):
    def appender(self,n):
        #print(n)
        self.outputFile="aggregate_{}.csv".format("full")
        drugcols=["primaryid","caseid","prod_ai"]
        democols=["primaryid","caseid","age","sex","wt","wt_cod"]
        reaccols=["primaryid","caseid","pt"]
        
        demo,drug,react=self.demographics[n], self.drugs[n], self.reactions[n]
        try:
            d=pd.read_csv(drug, sep='$', lineterminator='\r', usecols=drugcols,).dropna().reset_index(drop=True)
        except:
            
            print("n:{}\n".format(n),self.drugs[n],"\n",pd.read_csv(drug, sep='$', lineterminator='\r').columns)
        try:
            r=pd.read_csv(react, sep='$', lineterminator='\r', usecols=reaccols,)
        except:

            print("n:{}\n".format(n),self.react[n],"\n",pd.read_csv(react, sep='$', lineterminator='\r').columns)
        try:
            dem=pd.read_csv(demo, sep='$', lineterminator='\r', usecols=democols,)

                
        
                
        except:
            
            try:
                dem=pd.read_csv(demo, sep='$', lineterminator='\r', 
                                usecols=["primaryid","caseid","age","gndr_cod","wt","wt_cod"])
                dem=dem.rename({"gndr_cod":"sex"},axis=1)
            except:
                print("n:{}\n".format(n),self.demographics[n],"\n",pd.read_csv(demo, sep='$', lineterminator='\r').columns)
                
        try:
            if not self.mode:
                a=dataSetup(demo=dem, drug=d, reac=r)
                df=a.clean()
                df=a.finalDF

                df.dropna().to_csv(self.outputFile, mode="a", index=False)
                
            if "pair" in self.mode or "single" in self.mode:
                print("Mean Encoder setup")
                a=setupDataMeanEncoding(demo=dem, drug=d, reac=r, mode=self.mode)
                df=a.clean()
                self.outputFile="aggregate_{}.csv".format(self.mode)
                df.dropna().to_csv(self.outputFile, mode="a", index=False)
                
                
            
            else:


                a = setupDataFilter(demo=dem, drug=d, reac=r, reactionType=self.mode)
                df = a.clean()
                self.outputFile="aggregate_{}.csv".format(self.mode)

                df.dropna().to_csv(self.outputFile, mode="a", index=False)
            
        except:
            print("failed for {}".format(n))
            
    def multiprocessor(self):
        # for i in self.mode:
            # self.currentMode=i
#             processes = [None] * 4
#             for i in range(4):
#                 processes[i] = multiprocessing.Process(target=self.appender, args=(i,))
#                 processes[i].start()
#             for i in range(4):
#                 processes[i].join()
            #pool=multiprocessing.Pool(processes=6)
            csvList=self.demographics
            for i in range(len(csvList)):
                self.appender(i)
            #result=pool.map(self.appender, range(len(csvList)))
            #pool.close()
            #pool.join()
