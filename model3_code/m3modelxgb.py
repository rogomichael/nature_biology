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
print("LOADING DATA, WAIT.......\n\n\n")

#Load Train and Test Excel files
train_df = pd.read_excel("m3train.xlsx")
test_df = pd.read_excel("m3test.xlsx")
X_train = train_df.iloc[:,3:] #Training
y_train = train_df.iloc[:,2:3]
#print("Training data\n\n\n\n", X_train)
#print("Training labels\n\n\n\n", y_train)


X_test = test_df.iloc[:,3:]
X_test_raw = test_df.iloc[:,3:]
y_test = test_df.iloc[:,2:3]
y_test_raw = test_df.iloc[:,2:3]


#Validation_set
valid= train_df[train_df['Year'] == 2021]
dvalid_data = valid.iloc[:,3:]
dvalid_label = valid.iloc[:,2:3]
features = train_df.columns
myfeatures = features[3:]

dvalid = xgb.DMatrix(dvalid_data, label=dvalid_label, missing=-999.0, feature_names=myfeatures) #validation set
#print("Validation data\n", dvalid_data)
#print("Validation label\n", dvalid_label)
#sys.exit()
#First we have to convert these values to XgbDMatrix
#Training set
dtrain = xgb.DMatrix(X_train, label=y_train, missing=-999.0, feature_names=myfeatures) # = xgb.DMatrix(y_train)
#Validation set
dtest = xgb.DMatrix(X_test, label=y_test, missing=-999.0, feature_names=myfeatures) # = xgb.DMatrix(y_test)
#Test set
X_test = xgb.DMatrix(X_test)
y_test = xgb.DMatrix(y_test)
print("Data Preparation succesful.....\n")
print("Moving on to do computation....\n")
#sys.exit()

#Get the labels and scale the data
label = dtrain.get_label()
#disable scaling
ratio = float(np.sum(label == 0)) / np.sum(label == 1)
base_params = {
	'verbosity': 0,
	'booster': 'gbtree',
	'objective': 'binary:logistic',
        'scale_pos_weight' : ratio,
        'max_delta_step': 1,
        'tree_method': 'gpu_hist',
        'eval_metrics': 'logloss'
}
params = { 
        'learning_rate': 0.09880726959534625, 
        'num_boost_round': 198.863880711211, 
        'max-depth': 5, 
        'gamma': 1.5604867919189752e-08, 
        'subsample': 0.6569083845212979, 
        'reg_alpha': 7.461449079675969e-06, 
        'reg_lambda': 8.072448854162746e-08, 
        'colsample_bytree': 0.9103345902075206, 
        'min_child_weight': 0, 
        'n_estimators': 446}


params.update(base_params)

xgb_cv = cv(dtrain=dtrain,
            params = params,
           nfold=5,
           num_boost_round=2000,
           early_stopping_rounds=550,
           metrics='logloss',
           as_pandas=True,
           seed=0,
           verbose_eval=1
           )
print("------------------------------------------------------------#\n")
print("-------------------Finished Cross-Validation-------------------\n")
print("------------------------------------------------------------#\n")

#Save the plot of the cross-validation graph
#get train_log_loss-mean/std

tmean = xgb_cv['train-logloss-mean']
testmean = xgb_cv['test-logloss-mean']
plt.plot(tmean, label = "Train")
plt.plot(testmean, label="Validation")
plt.title("Cross-Validation curve for logloss mean")
plt.xlabel("n_rounds")
plt.ylabel("logloss")
plt.legend(loc="upper right")
plt.savefig("model3_xgb_cv_plot.png", dpi=300)

#Rerun training with the best hyperparameter combination
print("Rerunning using the best trial params...")
print("params = {}".format(params))
bst = xgb.train(params, dtrain,
        num_boost_round=1000,
	#evals=[(dtrain, 'train'), (dvalid, 'valid_2020'), (dtest, 'test_2021')],
	evals=[(dtrain, 'train'), (dvalid, 'valid_2021'), (dtest, 'test_2021')],
        #n_tree_limit = bst.best_iteration,
	early_stopping_rounds=50)
print("Finished Training...")

#Check accuracy of the model
print("Checking the acurracy of the model...\n")
preds = bst.predict(dtest)
labels = dtest.get_label()
print(
    "error=%f"
    % (
        sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))
    )
)

#save model
bst.save_model('model3xgb.model')




