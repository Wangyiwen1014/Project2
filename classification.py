# classification
import pandas as pd
import time
from texttable import Texttable

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


col_names =['id','Gender','Age','Driving_License','Region_Code','Previously_Insured',
            'Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel'
            ,'Vintage','Response']

#data loading
train=pd.read_csv('train.csv')
train = train.iloc[1:]
test=pd.read_csv('test.csv')
test = test.iloc[1:]

#data preprocessing
mappings = {
    "Male":0,
    "Female":1}

train['Gender'] = train['Gender'].apply(lambda x: mappings[x])
test['Gender'] = test['Gender'].apply(lambda x: mappings[x])

mappings = {
    "< 1 Year":0,
    "1-2 Year":1,
    "> 2 Years":2}

train['Vehicle_Age'] = train['Vehicle_Age'].apply(lambda x: mappings[x])
test['Vehicle_Age'] = test['Vehicle_Age'].apply(lambda x: mappings[x])


mappings = {
    "No":0,
    "Yes":1}

train['Vehicle_Damage'] = train['Vehicle_Damage'].apply(lambda x: mappings[x])
test['Vehicle_Damage'] = test['Vehicle_Damage'].apply(lambda x: mappings[x])



feature_cols =['Gender','Age','Driving_License','Region_Code','Previously_Insured',
               'Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel',
               'Vintage']


X_train = train[feature_cols].values # Features
Y_train = train.Response.values # Target variable

X_test = test[feature_cols].values # Features
Y_test = test.Response.values # Target variable


#Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=29, min_samples_split=350)
start = time.perf_counter()
dt.fit(X_train,Y_train)
y_pred1 = dt.predict(X_test)
end = time.perf_counter()
t1=[end-start]
a1=accuracy_score(Y_test, y_pred1)
q1=[("DT",{"max_depth":29,"min_samples_split":350}, a1)]

# KNN
KNN=KNeighborsClassifier(n_neighbors=30,weights='uniform') #KNN
start = time.perf_counter()
KNN.fit(X_train,Y_train)
y_pred2=KNN.predict(X_test)
end = time.perf_counter()
t1.append(end-start)
a2=accuracy_score(Y_test,y_pred2)
q1.append(("K-NN",{"n_neighbors":30,"weights":'uniform'}, a2))

# MLP classifier
mlp = MLPClassifier(solver="adam", batch_size=1000, max_iter=500, beta_1=0.85,beta_2=0.7)
start = time.perf_counter()
mlp.fit(X_train, Y_train)
y_pred3=mlp.predict(X_test)
end = time.perf_counter()
t1.append(end-start)
a3=accuracy_score(Y_test, y_pred3)
q1.append(("mlp",{"solver":"adam", "batch_size":1000, "max_iter":500, 
                  "beta_1":0.85,"beta_2":0.7}, a3))

#AdaBoost classifier
abc = AdaBoostClassifier(base_estimator=None,learning_rate=1.0,n_estimators=50,algorithm='SAMME.R',random_state=None)
start = time.perf_counter()
abc.fit(X_train,Y_train)
y_pred4 = abc.predict(X_test)
end = time.perf_counter()
t1.append(end-start)
a4=accuracy_score(Y_test, y_pred4)
q1.append(("Adaboost",{"base_estimator":"None","learning_rate":1.0,"n_estimators":50,
                       "algorithm":'SAMME.R',"random_state":"None"}, a4))

#Bagging classifier
bc = BaggingClassifier(n_estimators=50,max_samples=150,max_features=10,bootstrap=True,n_jobs=-1)
start = time.perf_counter()
bc.fit(X_train,Y_train)
y_pred5 = bc.predict(X_test)
end = time.perf_counter()
t1.append(end-start)
a5=accuracy_score(Y_test, y_pred5)
q1.append(("Bagging",{"n_estimators":50,"max_samples":150,"max_features":10,
                      "bootstrap":"True","n_jobs":-1}, a5))

# table to results of different classifiers
table = Texttable()
table.add_rows([["classifier","parameters","accuracy","training time"],
                [q1[0][0],q1[0][1],q1[0][2],t1[0]],
                [q1[1][0],q1[1][1],q1[1][2],t1[1]],
                [q1[2][0],q1[2][1],q1[2][2],t1[2]],
                [q1[3][0],q1[3][1],q1[3][2],t1[3]],
                [q1[4][0],q1[4][1],q1[4][2],t1[4]], 
               ])

print(table.draw())