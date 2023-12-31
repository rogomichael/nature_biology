from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
from sklearn import metrics
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import cv
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter 
import os
import sys


path=os.getcwd()
#print(path)
#Load Train and Test Excel files
train_df = pd.read_excel("m2train.xlsx")
test_df = pd.read_excel("m2test.xlsx")

#Training
X_train = train_df.iloc[:,3:] 
y_train = train_df.iloc[:,2:3]
print("Training data\n\n\n\n", X_train)
print("Training labels\n\n\n\n", y_train)

#Test
X_test = test_df.iloc[:,3:]
X_test_raw = test_df.iloc[:,3:]
y_test = test_df.iloc[:,2:3]
y_test_raw = test_df.iloc[:,2:3]
dvalid=xgb.DMatrix(X_test, label=y_test)


##SKlearn Interface
params = {'verbosity': 0, 
          'booster': 'gbtree', 
          'objective': 'binary:logistic', 
          'scale_pos_weight': 3.6063414634146342, 
          'tree_method': 'gpu_hist', 
          'eval_metrics': 'logloss', 
          'learning_rate': 0.09997063322394877, 
          'num_boost_round': 176.7046076947795, 
          'max-depth': 2, 
          'gamma': 1.0252475829704333e-08, 
          'subsample': 0.2375445447870469, 
          'reg_alpha': 2.4354466692171654e-07, 
          'reg_lambda': 0.00017618665270871804, 
          'colsample_bytree': 0.688051412945261, 
          'min_child_weight': 0, 
          'n_estimators': 425
        }

clf = xgb.XGBClassifier( **params)
clf.fit(X_train, y_train, eval_set=[(X_test_raw, y_test_raw)])

#Predict
pred1 = clf.predict(X_test_raw, iteration_range=(0,1))
predthresh01 = (clf.predict_proba(X_test_raw)[:, 1] >= 0.1).astype(bool)
predthresh02 = (clf.predict_proba(X_test_raw)[:, 1] >= 0.2).astype(bool)
predthresh03 = (clf.predict_proba(X_test_raw)[:, 1] >= 0.3).astype(bool)
predthresh04 = (clf.predict_proba(X_test_raw)[:, 1] >= 0.4).astype(bool)
predthresh05 = (clf.predict_proba(X_test_raw)[:, 1] >= 0.5).astype(bool)
labels = dvalid.get_label()
#print("Error = %f" % (np.sum((pred1 > 0.5) != y_test_raw) / float(len(y_test_raw))))
#print("Error = %f" % (np.sum((pred2 > 0.5) != y_test_raw) / float(len(y_test_raw))))
print( "error at 0.1 threshold =%f" % (sum(1 for i in range(len(predthresh01)) if int(predthresh01[i] > 0.5) != labels[i])/ float(len(predthresh01))))
print( "error at 0.2 threshold =%f" % (sum(1 for i in range(len(predthresh02)) if int(predthresh02[i] > 0.5) != labels[i])/ float(len(predthresh02))))
print( "error at 0.3 threshold =%f" % (sum(1 for i in range(len(predthresh03)) if int(predthresh03[i] > 0.5) != labels[i])/ float(len(predthresh03))))
print( "error at 0.4 threshold =%f" % (sum(1 for i in range(len(predthresh04)) if int(predthresh04[i] > 0.5) != labels[i])/ float(len(predthresh04))))
print( "error at 0.5 threshold =%f" % (sum(1 for i in range(len(predthresh05)) if int(predthresh05[i] > 0.5) != labels[i])/ float(len(predthresh05))))

#Calculate ROC Values
#roc = roc_auc_score(y_train, clf.decision_function(X_test_raw))
#Use 0.5 default threshold
ytest=y_test.to_numpy()
#Printing
items = [0.1, 0.2, 0.3, 0.4, 0.5]
for i in items:
    roc=roc_auc_score(ytest, clf.predict_proba(X_test_raw)[:,1] >i)
    print("\nRoc values at threshold {} is {}...".format(i, roc))
    
#Generate Sklearn Confusion Matrix
def plot_confusion(threshold,num):
    IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
    cm=ConfusionMatrixDisplay.from_estimator(IC,ytest, threshold, normalize='pred',  values_format='.2%')
    cm.ax_.set_title("Confusion Matrix of Model 1 at {}".format(num))
    cm.figure_.savefig("confusion_matrix_thresh_{}.png".format(num), dpi=300)
#Plot threshold0.1
plot_confusion(predthresh01, 0.1)
plot_confusion(predthresh02, 0.2)
plot_confusion(predthresh03, 0.3)
plot_confusion(predthresh04, 0.4)
plot_confusion(predthresh05, 0.5)

#Save the model
import pickle
pickle.dump(clf, open("sklearn_temp.pkl", "wb"))

