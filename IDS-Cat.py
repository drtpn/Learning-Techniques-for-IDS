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
from sklearn import svm
from sklearn.neural_network import MLPClassifier

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)

df=pd.read_csv('./KDDTrain+.csv')

df['src_bytes']=pd.cut(df['src_bytes'],4,labels=['0','1','2','4'])
df['dst_bytes']=pd.cut(df['dst_bytes'],4,labels=['0','1','2','4'])
df['count']=pd.cut(df['count'],4,labels=['0','1','2','4'])

df['duration']=pd.cut(df['duration'],3,labels=['0','1','2'])
df['wrong_fragment']=pd.cut(df['wrong_fragment'],3,labels=['0','1','2'])
df['urgent']=pd.cut(df['urgent'],3,labels=['0','1','2'])
df['hot']=pd.cut(df['hot'],3,labels=['0','1','2'])
df['num_failed_logins']=pd.cut(df['num_failed_logins'],3,labels=['0','1','2'])
df['lnum_compromised']=pd.cut(df['lnum_compromised'],3,labels=['0','1','2'])
df['lnum_root']=pd.cut(df['lnum_root'],3,labels=['0','1','2'])
df['lnum_file_creations']=pd.cut(df['lnum_file_creations'],3,labels=['0','1','2'])
df['lnum_shells']=pd.cut(df['lnum_shells'],3,labels=['0','1','2'])
df['lnum_access_files']=pd.cut(df['lnum_access_files'],3,labels=['0','1','2'])
df['lnum_outbound_cmds']=pd.cut(df['lnum_outbound_cmds'],3,labels=['0','1','2'])
df['srv_count']=pd.cut(df['srv_count'],3,labels=['0','1','2'])
df['serror_rate']=pd.cut(df['serror_rate'],3,labels=['0','1','2'])
df['srv_serror_rate']=pd.cut(df['srv_serror_rate'],3,labels=['0','1','2'])
df['rerror_rate']=pd.cut(df['rerror_rate'],3,labels=['0','1','2'])
df['srv_rerror_rate']=pd.cut(df['srv_rerror_rate'],3,labels=['0','1','2'])
df['same_srv_rate']=pd.cut(df['same_srv_rate'],3,labels=['0','1','2'])
df['diff_srv_rate']=pd.cut(df['diff_srv_rate'],3,labels=['0','1','2'])
df['srv_diff_host_rate']=pd.cut(df['srv_diff_host_rate'],3,labels=['0','1','2'])

df['dst_host_count']=pd.cut(df['dst_host_count'],3,labels=['0','1','2'])
df['dst_host_srv_count']=pd.cut(df['dst_host_srv_count'],3,labels=['0','1','2'])
df['dst_host_same_srv_rate']=pd.cut(df['dst_host_same_srv_rate'],3,labels=['0','1','2'])
df['dst_host_diff_srv_rate']=pd.cut(df['dst_host_diff_srv_rate'],3,labels=['0','1','2'])
df['dst_host_same_src_port_rate']=pd.cut(df['dst_host_same_src_port_rate'],3,labels=['0','1','2'])
df['dst_host_srv_diff_host_rate']=pd.cut(df['dst_host_srv_diff_host_rate'],3,labels=['0','1','2'])
df['dst_host_serror_rate']=pd.cut(df['dst_host_serror_rate'],3,labels=['0','1','2'])
df['dst_host_srv_serror_rate']=pd.cut(df['dst_host_srv_serror_rate'],3,labels=['0','1','2'])
df['dst_host_rerror_rate']=pd.cut(df['dst_host_rerror_rate'],3,labels=['0','1','2'])
df['dst_host_srv_rerror_rate']=pd.cut(df['dst_host_srv_rerror_rate'],3,labels=['0','1','2'])

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