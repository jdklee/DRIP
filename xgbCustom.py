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


class xgbCustom(XGBClassifier):

    def __init__(self, df, featureThreshold, featureReductionThreshold, reductionMercyThreshold,
                 binary=False, target=False, test=False, gpu=-1):
        if binary:
            objective="binary:logistic"
        else:
            objective="multi:softprob"

        XGBClassifier.__init__(self,eval_metric="auc",objective=objective)
        # df=self.remove_duplicates(df)
        df=sklearn.utils.shuffle(df)
        df=df.reset_index(drop=True)
        self.binary=binary
        self.target=target
        self.test_size = 0.2
        label = df.label
        self.label = label
        print("length of label:", len(label))
        features = df.drop(["pt", "label"], axis=1)
        self.features = features
        print("length of features:", len(features))
        self.gpu=gpu

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(features, label, test_size=self.test_size, stratify=label)
        print("length of x_train: {}\n length of x_test:{}\n length of y_train:{}\n length of y_test:{}"\
              .format(len(self.X_train), len(self.X_test), len(self.y_train), len(self.y_test)))
        if test:
            self.X_test=self.X_test[:int(len(self.X_test)*test)]
            self.X_train=self.X_train[:int(len(self.X_train)*test)]
            self.y_test=self.y_test[:int(len(self.y_test)*test)]
            self.y_train = self.y_test[:int(len(self.y_train) * test)]
        if binary:
            objective="binary:logistic"
        else:
            objective="multi:softprob"
        self.model = XGBClassifier(eval_metric="auc", objective=objective)
        self.featureThreshold = featureThreshold
        self.featureReductionThreshold = featureReductionThreshold
        self.counter=0
        self.removedCols={}
        self.usedCols={}
        self.featureImportances={}
        self.test=test
        self.reduceMercy=0
        self.reductionMercyThreshold=reductionMercyThreshold

    def predict_proba(self, traindata):
        return self.model.predict_proba(traindata)
    def fit(self):
        print("length of columns:", len(self.X_train.columns))
        print("fitting")

        if self.gpu != -1:
            self.model.set_param({"updater":"grow_gpu",
                             "predictor":"gpu_predictor",
                             "tree_method":"gpu_hist"})

        self.model.fit(self.X_train, self.y_train)
        if self.binary:
            average="binary"
        else:
            average="macro"
        print("train results:")
        predictions=self.model.predict(self.X_train)
        accuracy = sklearn.metrics.accuracy_score(self.y_train, predictions)
        f1_score = sklearn.metrics.f1_score(self.y_train, predictions, average=average)
        precision = sklearn.metrics.precision_score(self.y_train, predictions, average=average)
        recall = sklearn.metrics.recall_score(self.y_train, predictions, average=average)
        confusion_matrix = sklearn.metrics.confusion_matrix(self.y_train, predictions)
        feature_importances=self.model.feature_importances_
        self.featureImportances[self.target]={col:feat for col,feat in\
                                              zip(self.features.columns,feature_importances)}

        self.featureImportances[self.target] = dict(sorted(self.featureImportances[self.target].items(),
                                                      key=lambda x: x[1], reverse=True))
        if self.binary:
            auc = sklearn.metrics.roc_auc_score(self.y_train, predictions)
        print("train Accuracy:".format(str(self.counter)), accuracy, "\n")
        print("train f1_score:".format(str(self.counter)), f1_score, "\n")
        print("train precision:".format(str(self.counter)), precision, "\n")
        print("train recall:".format(str(self.counter)), recall, "\n")
        if self.binary:
            print("train auc:".format(str(self.counter)), auc, "\n")
        print("train cm:".format(str(self.counter)), confusion_matrix, "\n")
        self.model.save_model("{}_xgb_model.json".format(self.target))
    def predict(self):
        print("predicting")
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def remove_duplicates(self,df):
        features=df.drop(["label","pt"],axis=1)
        to_drop=features[features.duplicated(keep="first")==True].index
        df.drop(list(to_drop), axis=0, inplace=True)
        print(len(to_drop),"rows dropped")
        return df



    def refineFeatures(self):
        print("refining features")
        importances = self.model.feature_importances_
        rid = np.argsort(importances)[::-1][:self.featureReductionThreshold]
        # removedCols=[]
        # for idx,i in enumerate(self.features.columns):
        #     if idx in rid:
        #         removedCols.append(i)
        removedCols=[i for idx, i in enumerate(self.features.columns) if idx in rid]
        self.removedCols[self.counter]=removedCols

        newCols = [i for idx, i in enumerate(list(self.features.columns)) if idx not in rid]
        self.usedCols[self.counter]=newCols

        newFeatures = self.features[newCols]
        print("feature refined to:", len(newCols))
        self.features = newFeatures
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(newFeatures, self.label, test_size=self.test_size)
    def eval(self,y_pred,mode=False):
        predictions = y_pred
        #check if predictions is probability or int
       #if type(predictions[0]) != int or type(predictions[0]) != float:
        #    predictions=[np.argmax(i) for i in predictions]

        # evaluate predictions

        if self.binary:
            average="binary"
        else:
            average="macro"
        accuracy = sklearn.metrics.accuracy_score(self.y_test, predictions)
        f1_score = sklearn.metrics.f1_score(self.y_test, predictions, average=average)
        precision = sklearn.metrics.precision_score(self.y_test, predictions, average=average)
        recall = sklearn.metrics.recall_score(self.y_test, predictions, average=average)
        confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, predictions)
        if self.binary:
            auc = sklearn.metrics.roc_auc_score(self.y_test, predictions)
        if not mode:
            print("round {} Accuracy:".format(str(self.counter)), accuracy,"\n")
            print("round {} f1_score:".format(str(self.counter)), f1_score,"\n")
            print("round {} precision:".format(str(self.counter)), precision,"\n")
            print("round {} recall:".format(str(self.counter)), recall,"\n")
            if self.binary:
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
            if self.binary:
                print("Final auc:".format(str(self.counter)), auc, "\n")
            print("Final cm:".format(str(self.counter)), confusion_matrix, "\n")
            importanceDict = {col: importance for col, importance in zip(list(self.features.columns),
                                                                         list(self.model.feature_importances_))}
            importanceDict = dict(sorted(importanceDict.items(), key=lambda x: x[1]), reverse=True)
            self.featureImportances["final"] = importanceDict
        if self.binary:
            return accuracy, auc, f1_score, precision, recall, confusion_matrix
        return accuracy,f1_score,precision,recall,confusion_matrix
    def run(self):
        self.fit()
        y_pred = self.predict()
        accuracy,f1_score,precision,recall,confusion_matrix=self.eval(y_pred=y_pred)
        self.currentAccuracy=accuracy
        if self.binary:
            self.model.save_model("/home/jdklee/models/{}_xgb_model.json".format(self.target))

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
            accuracy,f1_score,precision,recall,confusion_matrix=self.eval(y_pred=y_pred)
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
        space = {
            'max_depth': hp.choice('max_depth', range(5, 30, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),

        }
        # if self.gpu != -1:
        #     print("GPU USED!")
        #     space["updater"]="grow_gpu"
        #     space["predictor"]="gpu_predictor"
        #     space["tree_method"]="gpu_hist"

        trials = Trials()

        best_hyperparams = fmin(fn=self.objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=25,
                                trials=trials)
        if self.binary:
            print("For reaction {}".format(self.target))
        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)
        self.bestHyperparams=best_hyperparams
    #Tune and rerun for final results

    def optimize_run(self):
        self.hyperTune()
        params=self.bestHyperparams
        if self.binary:
            objective="binary:logistic"
        else:
            objective="multi:softprob"
        model=XGBClassifier(
                            eval_metric="auc",
                            n_estimators=params['n_estimators'],
                            learning_rate=params["learning_rate"],
                            max_depth=params['max_depth'],
                            gamma=params['gamma'],
                            min_child_weight=params['min_child_weight'],
                            subsample=params["subsample"],
                            colsample_bytree=params['colsample_bytree'])
        # if self.gpu != -1:
        #     model.set_param({"updater":"grow_gpu",
        #                      "predictor":"gpu_predictor",
        #                      "tree_method":"gpu_hist"})

        self.model=model

        self.fit()
        y_pred = self.predict()
        accuracy, auc, f1_score, precision, recall, confusion_matrix = self.eval(y_pred=y_pred, mode="final")
        self.model.save_model("{}_xgb_model_tuned.json".format(self.target))





    def objective(self,space):
        clf = XGBClassifier(
            n_estimators=space['n_estimators'],
            learning_rate=space["learning_rate"],
            max_depth=space['max_depth'],
            gamma=space['gamma'],
            min_child_weight=space['min_child_weight'],
            subsample=space["subsample"],
            colsample_bytree=space['colsample_bytree'])
            # n_estimators=space['n_estimators'],
            # #eval_metric="auc",if self.gpu != -1
            # max_depth=int(space['max_depth']),
            # gamma=space['gamma'],
            # reg_alpha=int(space['reg_alpha']),
            # min_child_weight=int(space['min_child_weight']),
            # colsample_bytree=int(space['colsample_bytree']),
            # num_class=2)
        # if self.gpu != -1:
        #     clf.set_param({"updater":"grow_gpu",
        #                      "predictor":"gpu_predictor",
        #                      "tree_method":"gpu_hist"})

        evaluation = [(self.X_train, self.y_train),(self.X_test, self.y_test)]

        clf.fit(self.X_train, self.y_train,
                #eval_metric="auc",
                eval_set=evaluation,
                early_stopping_rounds=10, verbose=True)

        pred = clf.predict(self.X_test)
        #prob=clf.predict_proba(self.X_test)
        predictions = pred

        # train_accuracy = cross_val_score(clf, self.X_test, self.y_test, scoring="accuracy", cv=5)
        # train_f1_score = cross_val_score(clf, self.X_test, self.y_test, scoring="f1", cv=5)
        # train_precision = cross_val_score(clf, self.X_test, self.y_test, scoring="precision", cv=5)
        # train_recall = cross_val_score(clf, self.X_test, self.y_test, scoring="recall", cv=5)

        # if self.binary:
        #     auc = cross_val_score(clf, self.X_test, self.y_test, scoring="auc", cv=5)
        # if self.binary:
        #     average="binary"
        # else:
        #     average="macro"
        # accuracy = sklearn.metrics.accuracy_score(self.y_test, predictions)
        # f1_score = sklearn.metrics.f1_score(self.y_test, predictions, average=average)
        # precision = sklearn.metrics.precision_score(self.y_test, predictions, average=average)
        # recall = sklearn.metrics.recall_score(self.y_test, predictions, average=average)
        # confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, predictions)
        # if self.binary:
        #     auc = sklearn.metrics.roc_auc_score(self.y_test, predictions, average=average)
        result_dict = sklearn.model_selection.cross_validate(a.model, X_test, y=y_test,
                                                             scoring=["accuracy", "roc_auc", "recall",
                                                                      "precision", "f1", "jaccard"],
                                                             cv=5, )
        result_dict = {k: np.mean(v) for k, v in result_dict.items()}
        print(result_dict)
        auc=result_dict["test_roc_auc"]

        # print(" Accuracy:", accuracy,"\n")
        # print("f1_score:", f1_score,"\n")
        # print("precision:", precision,"\n")
        # print("recall:", recall,"\n")
        # if self.binary:
        #     print("auc:", auc,"\n")
        # print("cm:", confusion_matrix,"\n")
        return {'loss': -auc, 'status': STATUS_OK}