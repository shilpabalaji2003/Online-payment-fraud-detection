#Online fraud detection using machine learning
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#importing the data set
data=pd.read_csv("data.csv")
data.head()

#checking if there are any null values
print(data.isnull().sum())

#checking the different types of transactions
print(data.type.value_counts())

#visualizing the types of transactions in our dataset
sns.countplot(x='type', data=data)

#visualizing type vs amount
sns.barplot(x='type', y='amount', data=data)

#checking correlation of features of the data with the isFraud column
correlation=data.corr()
print(correlation['isFraud'].sort_values(ascending=False))

#transform categorical variables into numerical variables
data['type']=data['type'].map({'CASH_OUT':1, 'PAYMENT':2, 'CASH_IN':3, 'TRANSFER':4, 'DEBIT':5})
print(data.head())

#splitting the data into dependent and independent variables
x=np.array(data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']])
y=np.array(data['isFraud'])

#split data into train and test data sets
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.1, random_state=42)

#train a machine learning model
model=DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#feed a transaction into the model to classify it as fraudulent(1) or non-fraudulent(0)
features=np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
if model.predict(features)==[1]:
    print("Fraudulent transaction")
else:
    print("Non-fraudulent transaction")