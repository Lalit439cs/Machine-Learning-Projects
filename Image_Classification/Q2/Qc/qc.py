import pandas as pnd
import numpy as nmp
from matplotlib import pyplot as pt
import cvxopt as cop
from time import time
from math import inf
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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

def binary_data(x,y,d1,d2):
    lx=[]
    ly=[]
    m=len(y)
    for i in range(m):
        if (y[i,0]==d1):
            lx.append(x[i])
            ly.append(1)
        elif (y[i,0]==d2):
            lx.append(x[i])
            ly.append(-1)
    xx=nmp.array(lx)
    yy=nmp.array(ly)
    yy=yy.reshape(-1,1)
    return xx,yy
    # design matrix , y vec


testing_path= sys.argv[2]
training_path = sys.argv[1]
training_path=training_path.strip()+'/train_data.pickle'
testing_path = testing_path.strip()+'/test_data.pickle'

x,y=whole_data(training_path)
xt,yt=whole_data(testing_path)
classes=5
gamma=0.001
d1,d2=9,0 #2,3
d1%=classes#4
d2%=classes#0
x0,y0 =binary_data(x,y,d1,d2)
xt0,yt0=binary_data(xt,yt,d1,d2)

yy0 =y0.reshape(-1)
yyt0 =yt0.reshape(-1)

t1=time()

#linear_libsvm
#sklearn.svm.SVC
# C =1.0
# kernel ='linear'#'rbf'
# gamma=0.001
# verbose =True
svm_main =SVC(C=1.0,kernel ='linear',verbose =True)
model = make_pipeline(StandardScaler(), svm_main)
# clf = make_pipeline(StandardScaler(), SVC(C=1.0,kernel ='linear',verbose =True))

model.fit(x0,yy0)
sv1=svm_main.support_
nsv=sv1.size
print("idx-")
print(sv1)
t=time()-t1
acc=model.score(xt0,yyt0)
b =svm_main.intercept_
print("b-")
print(b)
#print
print('linear_libsvm -')
print("total SV",nsv)
print("w computed and can be printed")
# print("b",b)
print("training time",t)
print("accuracy",acc*100)

#gaussian_libsvm
t1=time()
svm_main =SVC(C=1.0,kernel ='rbf',gamma =0.001,verbose =True)
model = make_pipeline(StandardScaler(), svm_main)

model.fit(x0,yy0)
sv2=svm_main.support_
nsv=sv2.size
print("idx-")
print(sv2)
t=time()-t1
acc=model.score(xt0,yyt0)
b =svm_main.intercept_
#print
print('gaussian_libsvm -')
print("total SV",nsv)
print("w cannot be computed")
print("b",b)
print("training time",t)
print("accuracy",acc*100)
# sys.exit()

#linear
sv1 =set(sv1)
#gaussian
sv2=set(sv2)
svs =sv1.intersection(sv2)
#match sv= len(svs)
print('match svs',len(svs))