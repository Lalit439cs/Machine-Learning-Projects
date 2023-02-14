import pandas as pnd
import numpy as nmp
from matplotlib import pyplot as pt
import cvxopt as cop
from time import time
from math import inf
import sys
# import matplotlib.image as mpimg

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

def linear_kernel(x,y):
    m=len(y)
    p_arr = (x*y) @ (x*y).T
    p= cop.matrix(p_arr,tc='d')                    #Pij=yi.yj.(xiT.xj)
    q=cop.matrix((nmp.ones((m,1))* -1),tc='d')
    h_arr=nmp.array((list(nmp.zeros(m)) + list(nmp.ones(m)))).reshape((2*m,1))
    g=cop.matrix(nmp.vstack((-1*nmp.identity(m), nmp.identity(m))),tc='d')
    h=cop.matrix(h_arr,tc='d')
    a=cop.matrix(y.T,tc='d')
    b=cop.matrix(nmp.array([[0]]),tc='d')
    soln = cop.solvers.qp(p, q, g, h, a, b)

    return nmp.array(soln['x'])

def linear_accuracy(x,y,w,b):
    right_count=0
    m=len(y)
    w=w.reshape(-1,1)
    result=nmp.zeros(m,dtype='int8')
    wtxb=x@w + b
    for i in range(m):
        if wtxb[i][0] >= 0 :
            result[i]=1
        else:
            result[i]= -1
        right_count +=(result[i]==y[i][0])
    accuracy=right_count *100 / m
    return accuracy,result

def linear_sv(alphas,x,y,compare_0 =(10**(-5.2))):
    m=len(y)
    sv=[]
    n=32*32*3
    w=nmp.zeros(n) # dimension of input-32*32*3 =3072 =n
    for i in range(m):
        if alphas[i]>compare_0 :
            sv.append(i)
            w+=alphas[i] *y[i,0]*(x[i])
    sv=nmp.array(sv)
    return w,sv

def b_linear(sv,x,y,w):
    bmin=inf
    bmax=-inf
    for i in sv:
        wx=nmp.dot(w,x[i])
        if y[i,0]==1:
            bmin=min(bmin,wx)
        else:
            bmax=max(bmax,wx)
    b=(bmax + bmin)*(-0.5)
    return b


testing_path= sys.argv[2]
training_path = sys.argv[1]
training_path=training_path.strip()+'/train_data.pickle'
testing_path = testing_path.strip()+'/test_data.pickle'

x,y=whole_data(training_path)
xt,yt=whole_data(testing_path)
classes=5
d1,d2=9,0 #2,3
d1%=classes#4
d2%=classes#0
x0,y0 =binary_data(x,y,d1,d2)
xt0,yt0=binary_data(xt,yt,d1,d2)

#model-linear
t1=time()
alphas=linear_kernel(x0,y0)
w,sv=linear_sv(alphas,x0,y0)
b=b_linear(sv,x0,y0,w)
nsv=len(sv)
t=time()-t1
acc,prediction=linear_accuracy(xt0,yt0,w,b)
#print
print("total SV -",nsv)
print("w computed and can be printed")
print(w)
print("b -",b)
print("training time -",t)
print("accuracy -",acc)

#plot-


#top 5 images plot
x=list(x0)
m =len(y0)
alp=[0 for k in range(m)]

for i in sv:
    alp[i]=alphas[i]

#reason
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

#norm
name="w_pic"+pg
w=(w-w.min())/(w.max()-w.min())
w=w.reshape(32,32,3)
imgplot = pt.imshow(w)
print(name)
pt.savefig(name)






