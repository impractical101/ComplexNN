#####################################################
import numpy as np
from  batch_NN import *
#from  simple_NN import *


def get_iris():
	f=f=open("../data/iris.txt")
	L=f.readlines()

	A=[]
	for l in L:
	    l=l.split(",")
	    #print l
	    a1=[]
	    for k in l:
		a1.append(float(k))
	    A.append(a1)


	A=np.array(A)
	X1=A[:,:4]
	Y1=A[:,4:]


	X=np.zeros(X1.shape+(2,))
	Y=np.zeros(Y1.shape+(2,))



	X[:,:,0]=X1
	Y[:,:,0]=Y1
	X[:,:,1]=0
	Y[:,:,1]=0

	return(X,Y)


 








"""
X,Y=get_iris()
NN=Neural_Network_Batch(n_layers=4,nodes=[4,3])    
NN.train(X,Y,batch_size=1,lr=0.0001,epochs=10000,shuffle=False)

y_out=NN.forward(X)
"""

"""

X,Y=get_iris()
NN=Neural_Network(n_layers=2,nodes=[4,3])    
while(True):
    idx=np.random.randint(0,120)
    x_t=X[idx]
    y_t=Y[idx]

    NN.train(x_t,y_t,lr=0.0001)
    y_out=NN.forward(x_t)

    error=np.mean(np.abs(y_out-y_t))
    #print(error,y_out,y_t)  
    print(error)  

"""




"""
NN=Neural_Network(n_layers=3,nodes=[2,3,1])    
while(True):
    i=np.random.randint(0,2)
    if(i==0):
	X=np.array([[1,0],[2,0],[3,0]])
	Y=np.array([[1,0]])
    else:
	X=np.array([[-1,0],[-2,0],[-3,0]])
	Y=np.array([[-1,0]])
	
	
    NN.train(X,Y,lr=0.00001)
    y_out=NN.forward(X)

    error=np.mean(np.abs(y_out-Y))
    print(error)  
"""

"""
#NN=Neural_Network(n_layers=2,nodes=[2,1])    
NN=Neural_Network_Batch(n_layers=2,nodes=[2,1])    
while(True):
    i=np.random.randint(0,4)
    if(i==0):
	X=np.array([[[1,0],[0,0]]])
	Y=np.array([[[1,0]]])
    elif(i==1):
	X=np.array([[[0,0],[1,0]]])
	Y=np.array([[[0,0]]])
    elif(i==2):
	X=np.array([[[1,0],[1,0]]])
	Y=np.array([[[1,0]]])
    	
    else:
	X=np.array([[[0,0],[0,0]]])
	Y=np.array([[[0,0]]])
	
    NN.train(X,Y,lr=0.1,batch_size=1)
    y_out=NN.forward(X)

    error=np.mean(np.abs(y_out-Y))
    print(error)  

"""




