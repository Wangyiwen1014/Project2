# logistic regression
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


#data loading
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
feature_cols =['Gender','Age','Region_Code','Previously_Insured',
               'Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel',
               'Vintage']
#data mapping
train_data['Gender'][train_data['Gender'] == 'Male'] = 0
train_data['Gender'][train_data['Gender'] == 'Female'] = 1
train_data['Vehicle_Age'][train_data['Vehicle_Age'] == '< 1 Year'] = 0
train_data['Vehicle_Age'][train_data['Vehicle_Age'] == '1-2 Year'] = 1
train_data['Vehicle_Age'][train_data['Vehicle_Age'] == '> 2 Years'] = 2
train_data['Vehicle_Damage'][train_data['Vehicle_Damage'] == 'No'] = 0
train_data['Vehicle_Damage'][train_data['Vehicle_Damage'] == 'Yes'] = 1

test_data['Gender'][test_data['Gender'] == 'Male'] = 0
test_data['Gender'][test_data['Gender'] == 'Female'] = 1
test_data['Vehicle_Age'][test_data['Vehicle_Age'] == '< 1 Year'] = 0
test_data['Vehicle_Age'][test_data['Vehicle_Age'] == '1-2 Year'] = 1
test_data['Vehicle_Age'][test_data['Vehicle_Age'] == '> 2 Years'] = 2
test_data['Vehicle_Damage'][test_data['Vehicle_Damage'] == 'No'] = 0
test_data['Vehicle_Damage'][test_data['Vehicle_Damage'] == 'Yes'] = 1



#Set outliers to mean in Annual Premium
train_data.Annual_Premium[train_data.Annual_Premium > 200000] = train_data.Annual_Premium.mean()
test_data.Annual_Premium[test_data.Annual_Premium > 200000] = train_data.Annual_Premium.mean()


X_train = train_data[feature_cols].values # Features
Y_train = train_data.Response.values # Target variable

X_test = test_data[feature_cols].values # Features
Y_test = test_data.Response.values # Target variable

#X_train, X_test, Y_train, Y_test = train_test_split(train_data.drop(['id','Response'], axis=1), train_data['Response'], test_size = 0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
test_pred = logreg.predict(X_test)
test_pred_proba = logreg.predict_proba(X_test)

print("accuracy of logreg.predict : ",accuracy_score(Y_test, test_pred))

