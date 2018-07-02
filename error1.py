import matplotlib.pyplot as plt
import numpy as np
"""
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


#commputes (z-t)^2
def Er_im(z,t):
     var=np.zeros((1,2))
     r=(z[0][0]-t[0][0])*(z[0][0]-t[0][0])-(z[0][1]-t[0][1])*(z[0][1]-t[0][1])
     im= 2*(z[0][0]-t[0][0])*(z[0][1]-t[0][1])
     var[0][0]=r
     var[0][1]=im
     return var



#computes z1*z2
def im_mul(z1,z2):
     z3=np.zeros(z1.shape)
     z3[0,0]=z1[0,0]*z2[0,0]-z1[0,1]*z2[0,1]
     z3[0,1]=z1[0,0]*z2[0,1]+z1[0,1]*z2[0,0]
     #print("z3",z3)
     return z3


def DE_DZ(z,t):
    var=np.zeros((1,2))
    r=2*(z[0][0]-t[0][0])
    im=2*(z[0][1]-t[0][1])
    var[0][0]=r
    var[0][1]=im
    return var
 
def Er_ours(z,t):
     e=np.mean(np.abs(z-t))
     return e    

#normalize it to unit vector of shape z=(1,2)
def normalise(z):
    z=z/1.0
    absolute=np.sqrt(np.sum((z*z)))
    z=z/absolute
    return(z)


#------------------

t=np.array([[-300,200]])

z_in=np.array([[30,-200]])
w_temp=np.array([[3,9]])
z_out=im_mul(z_in,w_temp)




Er=[Er_im(z_out,t)]
Er_mae=[Er_ours(z_out,t)]

num_iter=10000
lamda=0.0001
for i in range(num_iter):

   dedz=DE_DZ(z_out,t)
   del_w=im_mul(dedz,z_in)
   w_temp=w_temp-lamda*del_w


   z_out=im_mul(z_in,w_temp)
   Er.append(Er_im(z_out,t))
   Er_mae.append(Er_ours(z_out,t))


Er=np.array(Er)
Er_mae=np.array(Er_mae)

plt.figure("real -img")
plt.plot(Er[:,0,1])
plt.plot(Er[:,0,0])
plt.figure("ours error")
plt.plot(Er_mae)

plt.show()

print z_out



"""

z_out=np.array([[100,-200]])    
t=np.array([[-300,200]])

w_temp=np.array([[3,9]])
z_in=np.array([[3,2]])
z_out=im_mul(z_in,w_temp)




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

"""
