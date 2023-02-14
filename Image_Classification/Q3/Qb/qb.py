#libraries
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

from sklearn.metrics import ConfusionMatrixDisplay
import pickle

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


testing_path= sys.argv[2]
training_path = sys.argv[1]
training_path=training_path.strip()+'/train_data.pickle'
testing_path = testing_path.strip()+'/test_data.pickle'

x,y=whole_data(training_path)
xt,yt=whole_data(testing_path)
k=5

# print(data_all)
#save datas with pickle

#q3 algo,training
classes=5
gamma=0.001
print('multi-class training values-')

# sklearn
y0 =y.reshape(-1)
yt0 =yt.reshape(-1)
#gaussian_libsvm
t1=time()
svm_main =SVC(C=1.0,kernel ='rbf',gamma =0.001,verbose =True,tol=1e-5,decision_function_shape='ovo')#break_ties
model = make_pipeline(StandardScaler(), svm_main)
classifier=model.fit(x,y0)
t=time()-t1

classes=svm_main.classes_
print("classes-")
print(classes)
print('fit_status',svm_main.fit_status_)
nf =svm_main.n_features_in_
print('n_features_in_',nf)
nsv=svm_main.n_support_

acc=model.score(xt,yt0)
predt=model.predict(xt)
bs =svm_main.intercept_
#print
print('gaussian_libsvm -')
print("w cannot be computed")
print("b",bs)
print("training time",t)
print("test accuracy",acc*100)

# tacc=model.score(x,y0)
# print("train accuracy",tacc*100)
# print("total SV",nsv)
print('nsv',nsv)
# print('predictions-\n',predt)

#confusion matrix
nmp.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
print("sklearn multi model confusion matrix-")
titles_options = [
    ("Confusion matrix, without normalization", None),("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        xt,
        yt0,
        display_labels=classes,
        cmap=pt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    pt.savefig(title+".png")

pt.show(block=False)
#save parameters in pickle-params

# may be in seperate part with the help of pickle 
#confusion matrix