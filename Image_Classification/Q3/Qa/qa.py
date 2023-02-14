#libraries
import pandas as pnd
import numpy as nmp
from matplotlib import pyplot as pt
import cvxopt as cop
from time import time
from math import inf
import sys

from sklearn.metrics import ConfusionMatrixDisplay

#q2 part of instance
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

def multi_data(x,y,k=5):
    dic ={}
    dic_ix={}
    for i in range(k):#empty yk
        dic[i]=[]
        dic_ix[i]=[]
        # print(type(i),i)
    m =len(y)
    # print('data-')
    for i in range(m):
        d=int(y[i,0]) #'numpy.uint8' key also fine
        # print(type(d),d)
        dic[d].append(x[i])
        dic_ix[d].append(i)
    return dic,dic_ix

def binary_data(dic,dic_ix,d1,d2):
    lx=[]
    ly=[]
    li=[]
    n1 =len(dic[d1])
    n2 =len(dic[d2])
    lx =dic[d1]+dic[d2]
    li=dic_ix[d1]+dic_ix[d2]
    ly =[1 for i in range(n1)]+[-1 for j in range(n2)]
    xx=nmp.array(lx)
    yy=nmp.array(ly)
    yy=yy.reshape(-1,1)
    ix =nmp.array(li)
    return xx,yy,ix        #may random shuffle but care corresponding indices as useful in sv
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
    return accuracy,result#prediction

testing_path= sys.argv[2]
training_path = sys.argv[1]
training_path=training_path.strip()+'/train_data.pickle'
testing_path = testing_path.strip()+'/test_data.pickle'

x,y=whole_data(training_path)
xt,yt=whole_data(testing_path)
k=5
data_multi,ix_multi= multi_data(x,y,k)
data_all ={}
pairs=[]
#pair of digit (i,j) ,i>j
# key - str(i)+str(j) in dict
for i in range(k-1,0,-1):
    for j in range(i-1,-1,-1):
        dkey = str(i)+str(j)
        pairs.append(dkey)
        # print(dkey)
        tdic={}
        tdic['x'],tdic['y'],tdic['ix']=binary_data(data_multi,ix_multi,i,j)
        data_all[dkey]=tdic
# print(data_all)
#save datas with pickle

#q3 algo,training
classes=5
gamma=0.001
params={}
print('multi-class training values-')
for p in pairs:
    print('classifier -',p)
    t1=time()
    xp,yp=data_all[p]['x'],data_all[p]['y']
    km = k_matrix(xp,xp,gamma)
    alphas=gaussian_kernel(xp,yp,km)
    sv,alp=gaussian_sv(alphas,xp,yp)
    b=b_gaussian(sv,alp,xp,yp,km)
    nsv=len(sv)
    t=time()-t1
    #saving parameters -
    #alpha,b in params dict, xp,yp in data_all dict
    params[p]={"alphas":alp,"b":b}
    # print
    print("total SV",nsv)
    print("w cannot be computed")
    print("b",b)
    print("training time",t)
    # print("accuracy",acc)

#save parameters in pickle-params


#testing
kmt=k_matrix(x,xt,gamma)

#mapping models with digit d with all its occurences in (n,k) classifiers
maps={}
for i in range(k):
    maps[i]=[]
for p in pairs:
    d1=int(p[0])
    d2 =int(p[-1])
    maps[d1].append(p)
    maps[d2].append(p)


def multi_results(x,xt,yt,data_all,params,k=5,kmt=kmt):
    right_count=0
    m=len(yt)
    #big km matrix of size(train*test)
    # kmt=k_matrix(x,xt,gamma)
    predictions=[]
    for i in range(m):
        xi=xt[i]
        dic_wxb={}
        pred_count=[0 for j in range(k)]
        for p in pairs:
            wx=0
            #sv_ix
            alp=params[p]['alphas']
            ix =data_all[p]["ix"]
            yp=data_all[p]["y"]
            mp=len(alp)
            for z in range(mp):
                if(alp[z]!=0):#sv
                    ixz=ix[z]
                    wx+=alp[z]*yp[z]*kmt[ixz,i]
            dic_wxb[p]=wx+params[p]['b']

            #bin pred with wxb>0-1,-1
            d1=int(p[0])
            d2 =int(p[-1])
            if (dic_wxb[p]>=0):
                pred_count[d1]+=1
            else:
                pred_count[d2]+=1
        
        #multi pred
        max_val=max(pred_count)
        ans=0
        if (pred_count.count(max_val)==1):
            ans =pred_count.index(max_val)
        else:
            score=-1* nmp.inf
            for d in range(k):
                if (pred_count[d]==max_val):
                    dscore=0
                    #d's models in maps
                    #dscore = sum_dp(y*wxb)
                    for dp in maps[d]:
                        if(d==int(dp[0])):
                            dscore+=dic_wxb[dp]*(1)#y_pred
                        else:
                            dscore+=dic_wxb[dp]*(-1)#y_pred
                        
                        # y_wxb=dic_wxb[dp]*(-1)#y_pred
                        # if(y_wxb>=0)
                        #may decide score on +ve
                    if (dscore>score):
                        score =dscore
                        ans =d
        predictions.append(ans)
        #multi pred

        #accuracy
        right_count +=(predictions[i]==yt[i][0])
    accuracy=right_count *100 / m

    return accuracy,predictions

#results
accuracy,predictions=multi_results(x,xt,yt,data_all,params,k=5,kmt=kmt)
print('accuracy',accuracy)


def binary_prediction(alp,yi,b,x,y,km):#respective d's x,y
    right_count=0
    m=len(y)
    alp=alp.reshape(-1,1)
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

#another way is calculating km for each classifier like binary q2
#calc all's ,wtxb=wx + b
# dic_wxb ={}
# 57.12 on simeple wxb
#on absolute,pred rightly-accuracy 57.1


# #train accuracy
# #results
# kmt0=k_matrix(x,x,gamma)
# accuracy0,predictions0=multi_results(x,x,y,data_all,params,k=5,kmt=kmt0)
# print('train accuracy -',accuracy0)

# may be in seperate part with the help of pickle 

# sklearn
y0 =y.reshape(-1)
yt0 =yt.reshape(-1)

#confusion matrix
nmp.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
print("own multi model confusion matrix-")
titles_options = [
    ("Confusion matrix, without normalization", None),("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_predictions(
        yt0,
        predictions,
        display_labels=classes,
        cmap=pt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    pt.savefig(title+".png")

pt.show(block=False)