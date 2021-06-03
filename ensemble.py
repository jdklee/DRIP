from gcpReader import *
from xgbCustom import *
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
import os
"""
DASK: Automate model training. Distributed model training. for scaling training

Add single new adverse event... for ensemble
In NN, need to retrain everything
"""
class ensemble():
    def __init__(self):
        #self.test_data=test_data
        arr = os.listdir()

        self.reaction=self.list_blobs()
        #For interpretation, keep indexes of each models
        self.model_dict={}
        self.model_index={}
        self.predictions={}
        self.evaluations={}
        self.feature_importances={}
        self.counter=0
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/jdklee/gcp/patentcitation-291203-3875e284f934.json'


    #Read models
    #Model_dict contains models for each reaction
    #Model index contains the mapping for eval
    def read_model(self, target_reaction):
        model_path="/home/jdklee/gcp/{}_xgb_model.json".format(target_reaction)
        if os.path.exists(model_path):
            a=XGBClassifier()
            a.load_model(model_path)
            self.model_index[self.counter]=target_reaction
            self.model_dict[target_reaction]=a
            self.counter+=1

    def list_blobs(self):
        """Lists all the blobs in the bucket."""
        from google.cloud import storage
        bucket_name = "abbvie_data_one"

        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(bucket_name)

        files = [blob.name for blob in blobs]
        files = [i.split("One")[0] for i in files if "aggregate" not in i]
        self.reactions=files
        #print(files)
        return files

    def save_test_data(self, x_test, y_test):
        x_test.to_csv("test_data_features_aggregate.csv", mode="a",index=False, header=True)
        y_test.to_csv("test_data_label_aggregate.csv", mode="a",index=False, header=True)



    def train(self):
        #Get all one hot encoded data from bucket
        reactions=self.reactions

        # Train separate models
        for reaction in reactions:
            #print(os.getcwd())
            path="/home/jdklee/{}_xgb_model.json".format(reaction)
            #print(path)
            if os.path.exists(path):
                print("model exists for ", reaction)
                continue
            else:
                print("model dont exist for {}, start training".format(reaction))
            try:
                self.model_index[reaction] = self.counter

                self.counter += 1
                a = onehotReader(aiDictPath="gs://abbvie_data_one/aggregate_full.csv",
                                 binary=True, target=reaction)
                df = a.read()
                a=xgbCustom(df=df, featureThreshold=1000,
                            featureReductionThreshold=500, reductionMercyThreshold=5,
                            binary=True, target=reaction)
                a.fit()

                #Hypertuning incoming!! ####
                ###########################


                self.model_dict[reaction] = a

                X_test=a.X_test
                y_test=a.y_test
                self.save_test_data(X_test,y_test)
                self.feature_importances[reaction]=a.featureImportances

                #Eval one model at a time, saving CM
                #model = self.model_dict[reaction]
                probability=a.predict_proba(X_test)
                print("target {} probability:".format(reaction), probability)
                self.predictions[reaction] = probability

                #Get CM
                print("calculating metrics results")
                from sklearn.model_selection import cross_val_predict
                from sklearn.metrics import confusion_matrix
                y_pred = cross_val_predict(a.model, X_test, y_test, cv=5)
                import sklearn
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)



                #crossvalidate results to get scores
                import sklearn
                #Get scores
                print(" ---getting cv scores---")
                result_dict=sklearn.model_selection.cross_validate(a.model, X_test, y=y_test,
                                                       scoring=["accuracy","roc_auc","recall",
                                                                "precision", "f1","jaccard"],
                                                                   cv=5,)
                result_dict = {k: np.mean(v) for k,v in result_dict.items()}
                self.evaluations[reaction] = {"confusion_matrix": confusion_matrix.tolist(),
                                              "result_dict": result_dict}
                print("for reaction {}, the cm is: \n".format(reaction), confusion_matrix)
                print("real results:\n",result_dict)

                #Dump results

                print("saving results")
                self.save_results(reaction)
            except Exception as e:
                print(e)
                print(reaction,"avoided")
                continue

        gcs = storage.Client()
        gcs.get_bucket('abbvie_data_one').blob("results.txt").upload_from_filename \
            ("/home/jdklee/result_dictionaries.txt".format(self.mode),
             content_type='text/csv')
        gcs.get_bucket('abbvie_data_one').blob("test_data_features_aggregate.csv").upload_from_filename \
            ("/home/jdklee/test_data_features_aggregate".format(self.mode),
             content_type='text/csv')
        gcs.get_bucket('abbvie_data_one').blob("test_data_label_aggregate.csv").upload_from_filename \
            ("/home/jdklee/test_data_label_aggregate.csv".format(self.mode),
             content_type='text/csv')



        #Upload full results to buvket



    def save_results(self, reaction):
        import json
        import json


        a={reaction: {"model_index":self.model_index[reaction], #takes in reaction
                "model_predictions":self.predictions[reaction].tolist(), #Takes in reaction
                "model_evaluations":self.evaluations[reaction],
                "feature_importances":self.feature_importances[reaction]}} #takes in reactio
        #print(a)
        print(self.evaluations[reaction])
        #print(self.feature_importances[reaction])
        with open("/home/jdklee/result_dictionaries.txt", "a") as f:
            f.write(str(a))

    def eval(self):
        #Get probabilities for each reaction
        for reaction in self.reactions:
            data=self.get_data(reaction)
            data = sklearn.utils.shuffle(data)

            features=data.drop(["label"], axis=1)
            if "pt" in features.columns:
                features.drop(["pt"],axis=1, inplace=True)
            labels=data[["label"]]

            X_train, X_test, y_train, y_test = \
                train_test_split(features, labels, test_size=0.33, stratify=labels)



            self.read_model(reaction)
            model=self.model_dict[reaction]
            probability=model.predict_proba(X_test)
            print("target {} probability:".format(reaction), probability)
            self.predictions[reaction]=probability

            # Eval with label
            tp=0
            fp=0
            tn=0
            fn=0
            for index, point in enumerate(probability):
                result=np.argmax(point)
                label=y_test[index]
                if label==1 and label==result:
                    tp+=1
                if label==0 and label==result:
                    tn+=1
                if label==1 and label!=result:
                    fn+=1
                if label==0 and label!=result:
                    fp+=1
            #create CM and add to dict
            confusion_matrix=[[tp, fp],
                              [fn, tn]]
            self.evaluations[reaction]=confusion_matrix
            print("for reaction {}, the cm is: \n".format(reaction), confusion_matrix)

    def get_data(self, target):
        a = onehotReader(aiDictPath="gs://abbvie_data_one/aggregate_full.csv", binary=True, target=target)
        df = a.read()
        return df

























