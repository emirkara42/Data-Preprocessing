#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('Data.csv')
#iloc->index locaiton[rows,columns](:)->range
#values->returns a numpy representation of the dataframe
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values #takes only last column
#print(X)
#print(y)

#Finding the Missing Data
missing_data = dataset.isnull().sum()
#print(missing_data)

#Taking Care of Missing Data(Mean or Median or Most Frequent etc.)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") #nan(NA) values->mean
imputer.fit(X[:,1:3]) #fitting
X[:,1:3] = imputer.transform(X[:,1:3]) #imputing
#print(X)

#Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder #To encode country names
#We connot give just 0-1-2 because the model may misintepret the data(numerical order)
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough") #kind of transformation, kind of encoding, indexes of the columns
#reminder="passthrough"-> to keep the other columns
X = np.array(ct.fit_transform(X))
#print(X)
#France->100 Spain->001 Germany->010
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) #no need to transform nparray
#print(y)

#Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#test_size=%20 of the features, random_state=1->no randomization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#print(X_train)
#print(X_test)
#print(y_train)
#print(X_test)

#Feature Scaling(After the split->scaling with train+test would cause imformation leakage on the test set)
#To prevent some features from dominating
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])#[:,:3]->Do not standartize the dummy variabels
X_test[:,3:] = sc.transform(X_test[:,3:])#same scalar from train set so just transform
#print(X_train)
#print(X_test)