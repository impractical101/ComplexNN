import matplotlib.pyplot as plt
import numpy as np
"""
def imgmul(y):
         
           z3=np.zeros(2,dtype=int)
           z3[0,0]=y[0,0]*y[0,0]+y[1,1]*y[1,1]
           z3[0,1]=y[0,0]*y[0,1]+z1[0,1]*z2[0,0]
           return z3
      	   
def E(x):
    return x*x

def DE_Dx(x):
    return 2*x


x=10

Er=[E(x)]
num_iter=1000
lamda=0.001
for i in range(num_iter):
   x=x-lamda*DE_Dx(x)
   e=E(x)
   Er.append(e)
plt.plot(Er)
"""


#input z=(x+iy) ->(1,2) gt(t)=(1,2)
def Er_im(z,t):
     var=np.zeros((1,2))
     r=(z[0][0]-t[0][0])*(z[0][0]-t[0][0])-(z[0][1]-t[0][1])*(z[0][1]-t[0][1])
     im= 2*(z[0][0]-t[0][0])*(z[0][1]-t[0][1])
     var[0][0]=r
     var[0][1]=im
     return var


def Er_ours(z,t):
     e=np.mean(np.abs(z-t))
     return e

def DE_DZ(z,t):
    var=np.zeros((1,2))
    r=2*(z[0][0]-t[0][0])
    im=2*(z[0][1]-t[0][1])
    var[0][0]=r
    var[0][1]=im
    return var


z=np.array([[100,-200]])    
t=np.array([[3,2]])


Er=[Er_im(z,t)]
Er_mse=[Er_ours(z,t)]

num_iter=10000
lamda=0.001
for i in range(num_iter):
   z=z-lamda*DE_DZ(z,t)
   e=Er_im(z,t)
   Er.append(e)
   Er_mse.append(Er_ours(z,t))


Er=np.array(Er)
Er_mse=np.array(Er_mse)

plt.figure("real -img")
plt.plot(Er[:,0,1])
plt.plot(Er[:,0,0])
plt.figure("ours error")
plt.plot(Er_mse)

plt.show()
