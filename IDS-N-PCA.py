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
from sklearn.decomposition import PCA

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
nn=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
pca=PCA(n_components=2)

df=pd.read_csv('./KDDTrain+.csv')

le=LabelEncoder()
le.fit(df['protocol_type'])
df['protocol_type']=le.transform(df['protocol_type'])

le.fit(df['service'])
df['service']=le.transform(df['service'])

le.fit(df['flag'])
df['flag']=le.transform(df['flag'])

le.fit(df['label'])
df['label']=le.transform(df['label'])

x=df.drop("label", axis=1)
y=df["label"]

pca.fit(x)
x=pca.transform(x)

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

#print(classification_report(Y_test,y_pred))
#print(confusion_matrix(Y_test,y_pred))