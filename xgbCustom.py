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
import sklearn
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


class xgbCustom():

    def __init__(self, df, featureThreshold, featureReductionThreshold, reductionMercyThreshold, test=False):
        self.seed = 7
        self.test_size = 0.33
        label = df.label
        self.label = label

        features = df.drop(["pt", "label"], axis=1)
        self.features = features

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(features, label, test_size=self.test_size, random_state=self.seed, shuffle=True)
        if test:
            self.X_test=self.X_test[:int(len(self.X_test)*test)]
            self.X_train=self.X_train[:int(len(self.X_train)*test)]
            self.y_test=self.y_test[:int(len(self.y_test)*test)]
            self.y_train = self.y_test[:int(len(self.y_train) * test)]

        self.model = XGBClassifier(objective="multi:softprob", num_classes=len(label.unique()))
        self.featureThreshold = featureThreshold
        self.featureReductionThreshold = featureReductionThreshold
        self.counter=0
        self.removedCols={}
        self.usedCols={}
        self.featureImportances={}
        self.test=test
        self.reduceMercy=0
        self.reductionMercyThreshold=reductionMercyThreshold
    def fit(self):
        print("fitting")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        print("predicting")
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def refineFeatures(self):
        print("refining features")
        importances = self.model.feature_importances_
        rid = np.argsort(importances)[:self.featureReductionThreshold]
        removedCols=[i for idx, i in enumerate(list(self.features.columns)) if idx in rid]
        self.removedCols[self.counter]=removedCols

        newCols = [i for idx, i in enumerate(list(self.features.columns)) if idx not in rid]
        self.usedCols[self.counter]=newCols

        newFeatures = self.features[newCols]
        self.features = newFeatures
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(newFeatures, self.label, test_size=self.test_size, random_state=self.seed)
    def eval(self,y_pred,mode=False):
        predictions = [np.argmax(value) for value in y_pred]
        # evaluate predictions
        accuracy = sklearn.metrics.accuracy_score(self.y_test, predictions)
        f1_score = sklearn.metrics.f1_score(self.y_test, predictions, average="macro")
        precision = sklearn.metrics.precision_score(self.y_test, predictions, average="macro")
        recall = sklearn.metrics.recall_score(self.y_test, predictions, average="macro")
        confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, predictions)
        auc = sklearn.metrics.roc_auc_score(self.y_test, predictions, average="macro")
        if not mode:
            print("round {} Accuracy:".format(str(self.counter)), accuracy,"\n")
            print("round {} f1_score:".format(str(self.counter)), f1_score,"\n")
            print("round {} precision:".format(str(self.counter)), precision,"\n")
            print("round {} recall:".format(str(self.counter)), recall,"\n")
            print("round {} auc:".format(str(self.counter)), auc,"\n")
            print("round {} cm:".format(str(self.counter)), confusion_matrix,"\n")
            importanceDict = {col: importance for col, importance in zip(list(self.features.columns),
                                                                         list(self.model.feature_importances_))}
            importanceDict=dict(sorted(importanceDict.items(), key=lambda x: x[1]), reverse=True)
            self.featureImportances[self.counter]=importanceDict
        if mode:
            print("Final Accuracy:".format(str(self.counter)), accuracy, "\n")
            print("Final f1_score:".format(str(self.counter)), f1_score, "\n")
            print("Final precision:".format(str(self.counter)), precision, "\n")
            print("Final recall:".format(str(self.counter)), recall, "\n")
            print("Final auc:".format(str(self.counter)), auc, "\n")
            print("Final cm:".format(str(self.counter)), confusion_matrix, "\n")
            importanceDict = {col: importance for col, importance in zip(list(self.features.columns),
                                                                         list(self.model.feature_importances_))}
            importanceDict = dict(sorted(importanceDict.items(), key=lambda x: x[1]), reverse=True)
            self.featureImportances["final"] = importanceDict
        return accuracy,f1_score,precision,recall,auc,confusion_matrix
    def run(self):
        self.fit()
        y_pred = self.predict()
        accuracy,f1_score,precision,recall,auc,confusion_matrix=self.eval(y_pred=y_pred)
        self.currentAccuracy=accuracy
        # predictions = [np.argmax(value) for value in y_pred]
        # # evaluate predictions
        # accuracy = sklearn.metrics.accuracy_score(self.y_test, predictions)
        # f1_score = sklearn.metrics.f1_score(self.y_test, predictions, average="macro")
        # precision = sklearn.metrics.precision_score(self.y_test, predictions, average="macro")
        # recall = sklearn.metrics.recall_score(self.y_test, predictions, average="macro")
        # confusion_matrix=sklearn.metrics.confusion_matrix(self.y_test, predictions)
        # auc = sklearn.metrics.roc_auc_score(self.y_test, predictions, average="macro")
        # print("round 1 Accuracy:", accuracy)
        # print("round 1 f1_score:", f1_score)
        # print("round 1 precision:", precision)
        # print("round 1 recall:", recall)
        # print("round 1 auc:", auc)
        # print("round 1:", confusion_matrix)
        # importanceDict = {col: importance for col, importance in zip(list(self.features.columns),
        #                                                              list(self.model.feature_importances_))}
        # importanceDict=dict(sorted(importanceDict.items(), key=lambda x: x[1]), reverse=True)
        # self.featureImportances[self.counter]=importanceDict
    def run_newfeatures(self):
        while len(self.X_train.columns) > self.featureThreshold and self.reduceMercy <= self.reductionMercyThreshold:
            self.counter += 1
            self.refineFeatures()
            self.fit()
            y_pred = self.predict()
            accuracy,f1_score,precision,recall,auc,confusion_matrix=self.eval(y_pred=y_pred)
            if accuracy>self.currentAccuracy:
                print("cut {} improved accuracy".format(str(self.counter)))
                self.currentAccuracy=accuracy
            if accuracy <= self.currentAccuracy:
                print("cut {} didn't improve accuracy".format(str(self.counter)))
                self.reduceMercy+=1
            # predictions = [round(value) for value in y_pred]
            # # evaluate predictions
            # accuracy = sklearn.metrics.accuracy_score(self.y_test, predictions)
            # f1_score = sklearn.metrics.f1_score(self.y_test, predictions, average="macro")
            # precision = sklearn.metrics.precision_score(self.y_test, predictions, average="macro")
            # recall = sklearn.metrics.recall_score(self.y_test, predictions, average="macro")
            # confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, predictions)
            # auc=sklearn.metrics.roc_auc_score(self.y_test, predictions, average="macro")
            # print("round {} Accuracy:".format(str(self.counter)), accuracy)
            # print("round {} f1_score:".format(str(self.counter)), f1_score)
            # print("round {} precision:".format(str(self.counter)), precision)
            # print("round {} recall:".format(str(self.counter)), recall)
            # print("round {} auc:".format(str(self.counter)), auc)
            # print("round {} cm:".format(str(self.counter)), confusion_matrix)
            # importanceDict = {col: importance for col, importance in zip(list(self.features.columns),
            #                                                              list(self.model.feature_importances_))}
            # importanceDict = dict(sorted(importanceDict.items(), key=lambda x: x[1]), reverse=True)
            # self.featureImportances[self.counter] = importanceDict
        self.printFeatureImportance()

    def printFeatureImportance(self):
        importanceDict={col:importance for col,importance in zip(list(self.features.columns),
                                                                 list(self.model.feature_importances_))}
        print(importanceDict)
        plot_importance(self.model)

    def hyperTune(self):
        space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                 'gamma': hp.uniform('gamma', 1, 9),
                 'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                 'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                 'n_estimators': 180,
                 'seed': 0
                 }
        trials = Trials()

        best_hyperparams = fmin(fn=self.objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)
        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)
        self.bestHyperparams=best_hyperparams
    #Tune and rerun for final results
    def optimize_run(self):
        self.hyperTune()
        params=self.bestHyperparams
        model=XGBClassifier(objective="multi:softprob",
                            num_classes=len(self.label.unique()),
                            n_estimators=params['n_estimators'],
                            max_depth=params['max_depth'],
                            gamma=params['gamma'],
                            reg_alpha=params['reg_alpha'],
                            reg_lambda=params["reg_lambda"],
                            min_child_weight=params['min_child_weight'],
                            colsample_bytree=params['colsample_bytree'])
        self.model=model

        self.fit()
        y_pred = self.predict()
        accuracy, f1_score, precision, recall, auc, confusion_matrix = self.eval(y_pred=y_pred, mode="final")





    def objective(self,space):
        clf = XGBClassifier(
            objective="multi:softprob", num_classes=len(self.label.unique()),
            reg_lambda=space["reg_lambda"],
            n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
            reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
            colsample_bytree=int(space['colsample_bytree']))

        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

        clf.fit(self.X_train, self.y_train,
                eval_set=evaluation, eval_metric="auc",
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(self.X_test)
        predictions = [round(value) for value in pred]

        accuracy = sklearn.metrics.accuracy_score(self.y_test, predictions)
        f1_score = sklearn.metrics.f1_score(self.y_test, predictions, average="macro")
        precision = sklearn.metrics.precision_score(self.y_test, predictions, average="macro")
        recall = sklearn.metrics.recall_score(self.y_test, predictions, average="macro")
        confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, predictions)
        auc = sklearn.metrics.roc_auc_score(self.y_test, predictions, average="macro")


        print(" Accuracy:", accuracy,"\n")
        print("f1_score:", f1_score,"\n")
        print("precision:", precision,"\n")
        print("recall:", recall,"\n")
        print("auc:", auc,"\n")
        print("cm:", confusion_matrix,"\n")
        return {'loss': -accuracy, 'status': STATUS_OK}