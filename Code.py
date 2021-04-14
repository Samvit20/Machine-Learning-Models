import pandas as pd
import numpy as np

dataset2=pd.read_csv('dataset2.csv')
dataset2=dataset2.drop([61,829])

dataset2['Age'].replace(np.nan,0,inplace=True)
np.mean(dataset2['Age'])
dataset2['Age'].replace(0,23.74,inplace=True)

Sex_dum=pd.get_dummies(dataset2.Sex,prefix='Sex').iloc[:,1:]
Embarked_dum=pd.get_dummies(dataset2.Embarked,prefix='Embarked')
dataset2=pd.concat([dataset2,Sex_dum],axis=1)
dataset2=pd.concat([dataset2,Embarked_dum],axis=1)

dataset2=dataset2.drop("Sex",axis=1)
dataset2=dataset2.drop("Embarked",axis=1)

x=dataset2.iloc[:,[2,4,5,6,8,10,11,12,13]].values
y=dataset2.iloc[:,[1]].values
x=x.astype(float)

import statsmodels.api as st
x=np.append(arr=np.ones((889,1)).astype(int),values=x,axis=1)
x_o=x[:,[0,1,2,3,6,7,8,9]]
reg_OLS=st.OLS(endog=y, exog=x_o).fit()
reg_OLS.summary()

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x_o,y,test_size=0.2,random_state=23)

from sklearn import preprocessing
mms = preprocessing.MinMaxScaler(feature_range =(0, 1))
x_train_mms = mms.fit_transform(x_train)
x_test_mms = mms.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression as lr
cf=lr()
cf.fit(x_train_mms,y_train)
y_pre=cf.predict(x_test_mms)

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pre))


















