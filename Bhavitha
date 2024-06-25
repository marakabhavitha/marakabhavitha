#Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Import Data
df=pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv')
#Describe Data
df.head()
#Data Preprocessing
df.info()
df.describe()
#REMOVING MISSING VALUES
df=df.dropna()
df.info()
#Data Visualization
sns.pairplot(df,x_vars=['displacement','horsepower','weight','acceleration','mpg'], y_vars=['mpg']);
sns.regplot(x='displacement',y='mpg',data=df);
#Define Target Variable (Y) and Feature Variables (X)
df.columns
y=df['mpg']
y.shape
X=df[['displacement','horsepower','weight','acceleration']]
X.shape
X
#SCALING DATA
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)
X
pd.DataFrame(X).describe()
#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
#Linear Regression model
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train, y_train)
lr.intercept_
lr.coef_
#prediction
y_pred=lr.predict(X_test)
y_pred
#model evaluation
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score
mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
r2_score(y_test,y_pred)
