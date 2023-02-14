import pandas as pnd
import numpy as nmp
from matplotlib import pyplot as pt
import cvxopt as cop
from time import time
from math import inf
import sys

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


def k_matrix(x,z,gamma):
    mx=len(x)
    mz=len(z)
    km=nmp.zeros((mx,mz))
    for i in range(mx):                  #can reduce these loop by a numpy possible fn
        for j in range(mz):
            xz_diff=x[i]-z[j]
            km[i,j]=nmp.dot(xz_diff,xz_diff)

    return nmp.exp(-gamma * km)


def gaussian_sv(alphas,x,y,compare_0 =(10**(-5.2))):
    m=len(y)
    sv=[]
    alp=nmp.zeros(m)
    for i in range(m):
        if alphas[i]>compare_0 :
            sv.append(i)
            alp[i]=alphas[i]
            
    sv=nmp.array(sv)
    return sv,alp

def b_gaussian(sv,alp,x,y,km):
    bmin=inf
    bmax=-inf
    for i in sv:
        coef=(y*(km[:,i]).reshape(-1,1))
        wx=nmp.dot(alp,coef.reshape(1,-1)[0])
        if (y[i,0] == 1):
            bmin=min(bmin,wx)
        else:
            bmax=max(bmax,wx)
    b=(bmax + bmin)*(-0.5)
    return b

def gaussian_accuracy(alp,yi,b,x,y,km):
    right_count=0
    m=len(y)
    alp=alp.reshape(-1,1)
    print("la",len(alp))
    wx=nmp.sum(((alp*yi)*km),axis=0)
    result=nmp.zeros(m,dtype='int8')
    wtxb=wx + b
    for i in range(m):
        if wtxb[i] >= 0 :
            result[i]=1
        else:
            result[i]= -1
        right_count +=(result[i]==y[i][0])
    accuracy=right_count *100 / m
    return accuracy,result


def gaussian_kernel(x,y,km):
    m=len(y)
    p_arr =km *(y @ y.T)
    p= cop.matrix(p_arr,tc='d')                    #Pij=yi.yj.(phi(xi)T.phi(xj))
    q=cop.matrix((nmp.ones((m,1))* -1),tc='d')
    h_arr=nmp.array((list(nmp.zeros(m)) + list(nmp.ones(m)))).reshape((2*m,1))
    g=cop.matrix(nmp.vstack((-1*nmp.identity(m), nmp.identity(m))),tc='d')
    h=cop.matrix(h_arr,tc='d')
    a=cop.matrix(y.T,tc='d')
    b=cop.matrix(nmp.array([[0]]),tc='d')
    soln = cop.solvers.qp(p, q, g, h, a, b)

    return nmp.array(soln['x'])

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

t1=time()
km=k_matrix(x0,x0,gamma)
# print("lkm",len(km),len(km[0]))
alphas=gaussian_kernel(x0,y0,km)
sv,alp=gaussian_sv(alphas,x0,y0)
b=b_gaussian(sv,alp,x0,y0,km)
nsv=len(sv)
t=time()-t1
kmt=k_matrix(x0,xt0,gamma)
acc,_=gaussian_accuracy(alp,y0,b,xt0,yt0,kmt)
#print
print("total SV",nsv)
print("w cannot be computed")
print("b",b)
print("training time",t)
print("accuracy",acc)


#top 5 images plot
x=list(x0)
m =len(y0)
alp =list(alp.reshape(-1))
top5=[]
#plot top5[j] img
pg=".png"
for j in range(5):
    mx =max(alp)
    ix=alp.index(mx)
    name ="top_"+str(j)+pg
    top5.append(x[ix].reshape(32,32,3))
    imgplot = pt.imshow(top5[j])
    print(name,y0[ix])
    pt.savefig(name)
    del[alp[ix]]
    del[x[ix]]


#linear
# sv1 =set(sv)
#gaussian
# sv2=set(sv)
# svs =sv1.intersection(sv2)
#match sv= len(svs)