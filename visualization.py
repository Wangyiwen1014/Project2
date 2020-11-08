# visulization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.head() #View first 5 data rows
test.head()

train.describe() #see how the data is distributed,the maximums, the minimums, the mean, ...

train.info() #see what type of data each column includes


# Count plot of the number of customers insured
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train.Response)
plt.title('Number of customers Insured');

# Distribution of categorical features
plt.figure(figsize=(15,15))
plt.subplot(3,3,1)
sns.countplot(train.Gender)
plt.subplot(3,3,2)
sns.countplot(train.Driving_License)
plt.subplot(3,3,3)
sns.countplot(train.Region_Code)
plt.subplot(3,3,4)
sns.countplot(train.Previously_Insured)
plt.subplot(3,3,5)
sns.countplot(train.Vehicle_Age)
plt.subplot(3,3,6)
sns.countplot(train.Vehicle_Damage)
plt.subplot(3,3,7)
sns.countplot(train.Policy_Sales_Channel)


# Relations of categorical features and the target variable
plt.figure(figsize=(15,15))
plt.subplot(3,3,1)
sns.countplot(x="Gender", hue="Response", data=train)
plt.subplot(3,3,2)
sns.countplot(x="Driving_License", hue="Response", data=train)
plt.subplot(3,3,3)
sns.countplot(x="Region_Code", hue="Response", data=train)
plt.subplot(3,3,4)
sns.countplot(x="Previously_Insured", hue="Response", data=train)
plt.subplot(3,3,5)
sns.countplot(x="Vehicle_Age", hue="Response", data=train)
plt.subplot(3,3,6)
sns.countplot(x="Vehicle_Damage", hue="Response", data=train)
plt.subplot(3,3,7)
sns.countplot(x="Policy_Sales_Channel", hue="Response", data=train)


#Distribution of discrete numeric features
plt.figure(figsize=(24,5))
plt.subplot(1,4,1)
train.Age.hist(bins=80)
plt.title("Age Distribution")
plt.subplot(1,4,2)
train.Vintage.hist(bins=30)
plt.title("Vintage Distribution")
plt.subplot(1,4,3)
train.Region_Code.hist(bins=50)
plt.title("Region Code Distribution")
plt.subplot(1,4,4)
train.Policy_Sales_Channel.hist(bins=80)
plt.title("Policy Sales Channel Distribution")

#Relations of discrete numeric features and the target variable
plt.figure(figsize=(24,5))
plt.subplot(1,4,1)
train.groupby('Response').Age.hist(bins=80)
plt.title("Age Distribution")
plt.subplot(1,4,2)
train.groupby('Response').Vintage.hist(bins=30)
plt.title("Vintage Distribution")
plt.subplot(1,4,3)
train.groupby('Response').Region_Code.hist(bins=50)
plt.title("Region Code Distribution")
plt.subplot(1,4,4)
train.groupby('Response').Policy_Sales_Channel.hist(bins=80)
plt.title("Policy Sales Channel Distribution")

#Outlier detection of discrete numeric features
plt.figure(figsize=(24,15))
plt.subplot(3,4,1)
sns.stripplot(x='Response', y='Age', data=train, alpha=0.01, jitter=True);
plt.title("Age Distribution")
plt.subplot(3,4,2)
sns.stripplot(x='Response', y='Vintage', data=train, alpha=0.01, jitter=True);
plt.title("Vintage Distribution")
plt.subplot(3,4,3)
sns.stripplot(x='Response', y='Region_Code', data=train, alpha=0.01, jitter=True);
plt.title("Region Code Distribution")
plt.subplot(3,4,4)
sns.stripplot(x='Response', y='Policy_Sales_Channel', data=train, alpha=0.01, jitter=True);
plt.title("Policy Sales Channel Distribution")

#Distribution of continuous numeric feature
plt.figure(figsize=(6,5))
train.Annual_Premium.hist(bins=100)
plt.title("Annual_Premium Distribution")

#Relations of continuous numeric features and the target variable
plt.figure(figsize=(6,5))
train.groupby('Response').Annual_Premium.hist(bins=100)
plt.title("Annual_Premium Distribution")

#Outlier detection of continuous numeric features
plt.figure(figsize=(6,5))
sns.boxplot(y = 'Response', x = 'Annual_Premium', data = train, fliersize = 0, orient = 'h')
sns.stripplot(y = 'Response', x = 'Annual_Premium', data = train,linewidth = 0.6, orient = 'h')

# #show all the correlations between variables
plt.figure(figsize = (16,10))
sns.heatmap(train.corr(),linewidths=1,annot=True)
plt.plot()

# Correlation between Age and Response
def brac(x):
    if (x>=20) & (x<31):
        return '20-30'
    if(x>=31) & (x<41):
        return '31-40'
    if(x>=41) & (x<51):
        return '41-50'
    if(x>=51) & (x<61):
        return '51-60'
    if(x>=61) & (x<71):
        return '61-70'
    if(x>=71) & (x<81):
        return '71-80'
    if(x>=81) & (x<91):
        return '81-90'
    
train['AgeBracket']=train['Age'].apply(brac)
sns.countplot('AgeBracket',data=train,hue='Response')
plt.plot()
t1=pd.DataFrame(train.groupby(['AgeBracket'])['Response'].value_counts(normalize=True)*100)
t1

