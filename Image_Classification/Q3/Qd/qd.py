#libraries
import pandas as pnd
import numpy as nmp
from matplotlib import pyplot as pt
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle
import sys
#5-fold cross validation-
cross_k=5
Cvals =[1e-5,1e-3,1,5,10]
print('Cvals')
print(Cvals)
nc =len(Cvals)

# #train,test data
# with open('saved_data.pkl', 'rb') as f:
#     data_save = pickle.load(f)

# x=data_save['x']
# y0=data_save['y0']
# xt=data_save['xt']
# yt0=data_save['yt0']

#q2 part of instance -data 
def whole_data(path):
    df = pnd.read_pickle(path)
    Y=df["labels"]
    m =len(Y)
    X=df["data"]
    lx=[]
    for i in range(m):
        xi=X[i].reshape(-1)/255
        lx.append(xi)
    
    # X= X/255
    X=nmp.array(lx)
    return X,Y

testing_path= sys.argv[2]
training_path = sys.argv[1]
training_path=training_path.strip()+'/train_data.pickle'
testing_path = testing_path.strip()+'/test_data.pickle'


x,y=whole_data(training_path)
xt,yt=whole_data(testing_path)
k=5
classes=5
gamma =0.001
y0 =y.reshape(-1)
yt0 =yt.reshape(-1)

val_acc=[]
test_acc=[]
# models=[]
best_c=1.0
best_val=0
for i in range(nc):
    c=Cvals[i]
    print(c," related -")
    modl =SVC(C=c,kernel ='rbf',gamma =0.001,verbose =True,decision_function_shape='ovo')#break_ties
    scores=cross_val_score(modl, x, y0, cv=cross_k)
    vscore = nmp.mean(scores)
    val_acc.append(vscore)
    if (vscore>best_val):
        best_val=vscore
        best_c=c
    clf=make_pipeline(StandardScaler(), modl)
    clf.fit(x,y0)
    acct=clf.score(xt,yt0)
    test_acc.append(acct)

print('val_acc')
print(val_acc)

print('test_acc')
print(test_acc)  

print('best_val')
print(best_val)
# val_acc
# [0.37570000000000003, 0.37570000000000003, 0.58, 0.6079, 0.6093]
# best_val
# 0.6093
# best_c
# 10
print('best_c')
print(best_c)


###########
# plots

# plot 
# cvals,val_acc,test_acc
log_c=list(nmp.log(nmp.array(Cvals)))
print(log_c)
pt.figure(figsize=(8,8))
# markers ={'validation_accuracy':'x','test_accuracy':'o'}
# methods=["validation_accuracy",'test_accuracy']
pt.plot(log_c,test_acc, label ='test_accuracy',marker='o')
pt.plot(log_c,val_acc, label ='validation_accuracy',marker='x')

pt.title('Analysis of c parameter and k-fold validation-')
pt.xlabel('log(C)')
pt.ylabel('Accuracies- ')

pt.legend()
pt.savefig('analysis_c_kfold.png')
pt.show(block=False)
