
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
nn=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
df=pd.read_csv('./KDDTrain+.csv')

df2=df['label'].replace("back","DOS")
df2=df2.replace("land","DOS")
df2=df2.replace("smurf","DOS")
df2=df2.replace("pod","DOS")
df2=df2.replace("neptune","DOS")
df2=df2.replace("teardrop","DOS")

df2=df2.replace("ipsweep","probe")
df2=df2.replace("portsweep","probe")
df2=df2.replace("nmap","probe")
df2=df2.replace("satan","probe")

df2=df2.replace("loadmodule","u2r")
df2=df2.replace("rootkit","u2r")
df2=df2.replace("perl","u2r")
df2=df2.replace("buffer_overflow","u2r")

df2=df2.replace("guess_passwd","r2l")
df2=df2.replace("multihop","r2l")
df2=df2.replace("ftp_write","r2l")
df2=df2.replace("spy","r2l")
df2=df2.replace("phf","r2l")
df2=df2.replace("imap","r2l")
df2=df2.replace("warezmaster","r2l")
df2=df2.replace("warezclient","r2l")

'''
a=(df2 == 'DOS').sum()
b=(df2 == 'probe').sum()
c=(df2 == 'u2r').sum()
d=(df2 == 'r2l').sum()
e=(df2 == 'normal').sum()

print('DOS Attacks: ',(df2 == 'DOS').sum())
print('Probe Attacks: ',(df2 == 'probe').sum())
print('U2R Attacks: ',(df2 == 'u2r').sum())
print('R2L Attacks: ',(df2 == 'r2l').sum())

print('Attacks', a+b+c+d)
print('Normal', e)
print('Total', a+b+c+d+e)'''

le=LabelEncoder()
le.fit(df['protocol_type'])
df['protocol_type']=le.transform(df['protocol_type'])

le.fit(df['service'])
df['service']=le.transform(df['service'])

le.fit(df['flag'])
df['flag']=le.transform(df['flag'])

le.fit(df2)
df2=le.transform(df2)

x=df.drop("label", axis=1)
y=df2

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0,test_size=0.2)

lr.fit(X_train,Y_train)
y_pred1=lr.predict(X_test)
print('Logistic Regression: ', accuracy_score(Y_test,y_pred1))

dt.fit(X_train,Y_train)
y_pred3=dt.predict(X_test)
print('Decision Tree: ', accuracy_score(Y_test,y_pred3))

nn.fit(X_train,Y_train)
y_pred5=nn.predict(X_test)
print('Neural Network: ', accuracy_score(Y_test,y_pred5))

rf.fit(X_train,Y_train)
y_pred=rf.predict(X_test)
print('Random Forest: ', accuracy_score(Y_test,y_pred))

gbm.fit(X_train,Y_train)
y_pred2=gbm.predict(X_test)
print('Gradient Boosting: ', accuracy_score(Y_test,y_pred2))