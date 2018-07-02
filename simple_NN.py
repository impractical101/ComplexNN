#####################################################
import numpy as np
from  batch_NN import *


def get_iris():
	f=f=open("../data/iris.txt")
	L=f.readlines()

	A=[]
	for l in L:
	    l=l.split(",")
	    print l
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










class Neural_Network(object):
    def __init__(self,n_layers=5,nodes=[4,5,4,5,3]):
        
	"""
        n_layers=input("enter the number of layers including the output layer")
        self.l=int(n_layers)
        self.node_list=[]

        for i in range(self.l):
            n_nodes=int(input("enter the nodes in that layes:"))
            self.node_list.append(n_nodes)     
	"""  
        self.l=n_layers
        self.node_list=nodes
  
        self.W=[]
        self.ZL=[]   #output of each layer
        self.OL=[]  
        for i in range(len(self.node_list)-1):
            w_temp=np.random.random((self.node_list[i],self.node_list[i+1],2))
            self.W.append(w_temp)
        #print(self.W)    
        for n in self.node_list:
                                      
            z_temp=np.zeros((n,2))
            self.ZL.append(z_temp)
    def get_weights(self):
        return self.W
        
    #z1,z2 shape=(1,2)    
    def imgmul(self,z1,z2):
       # print(z1.shape)
        #print("\n")
        #print(z2.shape)
        #print("z1,z2",z1,z2)
        z3=np.zeros(z1.shape)
        z3[0,0]=z1[0,0]*z2[0,0]-z1[0,1]*z2[0,1]
        z3[0,1]=z1[0,0]*z2[0,1]+z1[0,1]*z2[0,0]
        #print("z3",z3)
        return z3
 
    #(batch_size,input_shape,2) 
    def forward(self, X):
        #forward propagation through our network
        self.ZL[0]=X
        for i in range(0,self.l-1,1):
            for j in range(0,self.node_list[i+1],1):
                t=np.zeros((1,2))
                for k in range(0,self.node_list[i],1):
                    t += self.imgmul(self.ZL[i][k:k+1,:],self.W[i][k][j:j+1,:]) ##
                    #print(self.ZL[i][k:k+1,:],self.W[i][k][j:j+1,:],t)
                self.ZL[i+1][j:j+1,:]=t
        #print("original",self.ZL)
        #print("end")
        return  self.ZL[-1]  

    
    #z is NN output y=gt
    def backward(self, X, y, z,lr=0.000001):
                self.o_error=np.zeros_like(z)
                self.o_error=2*(z-y)

                
                #self.o_error[:,0]=2*(z[:,0]-y[:,0])+2*(z[:,0]-y[:,1])
                #self.o_error[:,1]=2*(z[:,1]-y[:,0])+2*(z[:,1]-y[:,1])
		
   
	        #k1=np.mean(np.sqrt(self.o_error*self.o_error)) 
	
            
                px=np.zeros((1,2))
                
                tl=self.l-1
                while(tl>0):
                    DW=np.zeros((self.node_list[tl-1],self.node_list[tl],2))
                    for j in range(0,self.node_list[tl-1],1):
                        for k in range(0,self.node_list[tl],1):
                            #print(self.o_error[j:j+1].shape)
                            #print("values of j and k",j,k)
                            #print("before")
                            #print("input")
                            #print(self.ZL[1][0:1,:])
                            dw_jk=self.imgmul(self.ZL[tl-1][j:j+1,:],self.o_error[k:k+1])
                            #print("updates")
                            #print(dw_jk.shape)
			    #k1=np.mean(np.sqrt(dw_jk*dw_jk)) 
			    #k1=1

                            self.W[tl-1][j][k:k+1,:]-=lr*dw_jk    #update
                            DW[j][k]=self.imgmul(self.W[tl-1][j][k:k+1,:],self.o_error[k:k+1])

		    #self.o_error=2*self.o_error/(np.max(self.o_error)-np.min(self.o_error))
                    self.o_error=np.sum(DW,axis=1)##
                    #print("after",self.o_error)
	            #k1=np.mean(np.sqrt(self.o_error*self.o_error)) 
                    #self.o_error=self.o_error/k1

                    tl=tl-1
                    
                                               
    def train (self, X, y,lr=0.001):
            z = self.forward(X)
            self.backward(X, y, z,lr)




"""
X,Y=get_iris()
NN=Neural_Network(n_layers=3,nodes=[4,3,3])    
#NN=Neural_Network_Batch(n_layers=2,nodes=[4,3])    


while(True):
    idx=np.random.randint(0,120)
    #x_t=X[idx:idx+b]
    #y_t=Y[idx:idx+b]

    x_t=X[idx]
    y_t=Y[idx]

    #NN.train(x_t,y_t,lr=0.00000,batch_size=4)
    NN.train(x_t,y_t,lr=0.000001)
    y_out=NN.forward(x_t)

    error=np.mean(np.abs(y_out-y_t))
    print(error)  
    print(error,y_out,y_t)  

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
NN=Neural_Network(n_layers=3,nodes=[2,3,1])    
#NN=Neural_Network_Batch(n_layers=2,nodes=[2,1])    
while(True):
    i=np.random.randint(0,2)
    if(i==0):
	X=np.array([[1,0],[0,0]])
	Y=np.array([[1,0]])
    elif(i==1):
	X=np.array([[0,0],[1,0]])
	Y=np.array([[0,0]])
    elif(i==2):
	X=np.array([[1,0],[1,0]])
	Y=np.array([[0,0]])
    	
    else:
	X=np.array([[0,0],[0,0]])
	Y=np.array([[0,0]])
	
    NN.train(X,Y,lr=0.0001)
    y_out=NN.forward(X)

    error=np.mean(np.abs(y_out-Y))
    print(error,y_out,Y)  

"""




