#best_parameters_finding
import pandas as pd
from sklearn.model_selection import GridSearchCV
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

#add ‘.values' for ANN
X_train = train[feature_cols].values # Features
Y_train = train.Response.values # Target variable

X_test = test[feature_cols].values # Features
Y_test = test.Response.values # Target variable


# based on knn
model = KNeighborsClassifier(weights='uniform')
# Obtain the optimal model based on grid search
params = [{'n_neighbors':[20,25,30,35]}]
model = GridSearchCV(estimator=model, param_grid=params,scoring="accuracy",n_jobs=1,cv=10)	 
model.fit(X_train,Y_train)

print("best_parameter of the model：",model.best_params_)
print("best_score of the model：",model.best_score_)
print("best_estimator of the model：",model.best_estimator_)
