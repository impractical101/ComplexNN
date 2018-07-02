#####################################################
import numpy as np
import sys



class Neural_Network_Batch(object):
    def __init__(self,n_layers=5,nodes=[4,5,4,5,2]):

        """
        n_layers=input("enter the number of layers including the output layer")
        self.l=int(n_layers)
        self.node_list=[]

        for i in range(self.l):
            n_nodes=int(input("enter the nodes in that layes:"))
            self.node_list.append(n_nodes)     
        """
        self.node_list=[]
        self.l=n_layers
        self.node_list=nodes
        self.batch_size=1
        self.W=[]
        self.ZL=[]   #output of each layer
        self.OL=[]
        for i in range(len(self.node_list)-1):
            w_temp=np.random.random((self.node_list[i],self.node_list[i+1],2))
            self.W.append(w_temp)
        #print(self.W)   
        
        

        #print("mdo",self.ZL)
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
    
    #z3=(4,1,2) z4=(1,2)  return=(4,1,2)
    def batch_img_mul(self,z3,z4):
        b=z3.shape[0]
        
        z5=np.zeros(z3.shape)
        for i in range(0,b,1):
            z5[i]=self.imgmul(z3[i],z4)
      
        return z5    

    #z1=(b,1,2) z2=(b,1,2)  return=(b,1,2)
    def batch_img_mul1(self,z1,z2):
        b=z1.shape[0]
        
        z3=np.zeros(z1.shape)
        for i in range(0,b,1):
            z3[i]=self.imgmul(z1[i],z2[i])
      
        return z3    
    

    def forward(self, X,batch_size=None):
        #forward propagation through our network
        if(batch_size==None):
            b=X.shape[0]
        else:
            b=batch_size
	
        self.ZL=[]
        for n in self.node_list:
            z_temp=np.zeros((b,n,2))
            self.ZL.append(z_temp)
    
        self.ZL[0]=X
        for i in range(0,self.l-1,1):
            for j in range(0,self.node_list[i+1],1):
                t=np.zeros((b,1,2))
                for k in range(0,self.node_list[i],1):
                    #print(self.W[i].shape,self.ZL[i].shape)
                    t=t+self.batch_img_mul(self.ZL[i][:,k:k+1,:],self.W[i][k][j:j+1,:])
                
                #ZL[i+1].shape=(b,num_nodes_next_layer,2)
                self.ZL[i+1][:,j:j+1,:]=t

        return  self.ZL[-1]

    
    #X=(b,num_input_node,2)  y=(b,num_output_node,2) z=(b,num_output_node,2)  
    def backward(self, X, y, z,lr=0.000001):
        self.o_error=np.zeros_like(z)
        self.o_error=2*(y-z)
        LOSS=np.mean(np.abs(self.o_error),axis=(0,1))
        
         

        tl=self.l-1
        while(tl>0):
            DW=np.zeros((self.batch_size,self.node_list[tl-1],self.node_list[tl],2))
            cc=np.zeros((self.batch_size,self.node_list[tl-1],self.node_list[tl],2))
            for j in range(0,self.node_list[tl-1],1):
                for k in range(0,self.node_list[tl],1):

                    #print(self.o_error.shape,self.ZL[tl-1].shape)
                    dw_jk=self.batch_img_mul1(self.ZL[tl-1][:,j:j+1,:],self.o_error[:,k:k+1])  #returns (b,1,2)
                    DW[:,j,k:k+1,:]=dw_jk

                    cc[:,j,k:k+1,:]=self.batch_img_mul(self.o_error[:,k:k+1],self.W[tl-1][j][k:k+1,:])
                  
            new_DW=np.mean(DW,axis=0)
	    k1=np.mean(np.sqrt(new_DW*new_DW),axis=-1) 
	    k1=np.expand_dims(k1,-1)
            self.W[tl-1]-=lr*new_DW /k1  #update
                    

            #self.o_error=2*self.o_error/(np.max(self.o_error)-np.min(self.o_error))
            self.o_error=np.sum(cc,axis=2)##
            
            tl=tl-1
	  
	return LOSS[np.newaxis,:]

    #X=input (num_sample,dims)   Y=ground_truth (num_sample,dims_out)
    def train (self, X, Y,lr=0.0000001,batch_size=1,epochs=1,shuffle=True):
    	self.batch_size=batch_size
	"""
	Y_out = self.forward(X_temp,batch_size)
 	self.backward(X,Y Y_temp,,lr)
	"""
	
			
    	num_it=X.shape[0]/batch_size
	for e in range(epochs):
	    loss=np.zeros((1,2))
	    for i in range(num_it):
	    	idx=i*batch_size
		
		if(shuffle==True):
		    idx=np.random.randint(0,X.shape[0]-batch_size+1)

		X_temp=X[idx:idx+batch_size]
		Y_temp=Y[idx:idx+batch_size]
                #print Y_temp
	        Y_out = self.forward(X_temp,batch_size)
		l=self.backward(X, Y_out,Y_temp,lr)
	        loss=(loss+l)/2.0
	        sys.stdout.write("\r epochs:%d |  %d/%d  |  error: %0.2f + i %0.2f "%(e,i,num_it-1,loss[0][0],loss[0][1]))    	
            #print("\n") 


"""

NN=Neural_Network_Batch(n_layers=1,nodes=[3,2,1])
X=np.random.random((11000,3,2))
Y=np.random.random((11000,1,2))

NN.train(X,Y,batch_size=1,lr=0.001,epochs=10)
#Y_out=NN.forward(X)

"""





	




