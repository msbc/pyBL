import h5py
#from mayavi import mlab
from numpy import array
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import  matplotlib
import gc
#import psutil
from matplotlib import rc
from athena_read import athdf as athdf
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]

k1=int(raw_input('First:'))
k2=int(raw_input('Last:'))
f0=athdf('/perseus/scratch/gpfs/sashaph/BLayer/Mach9highresstampede/BL.out2.00000.athdf')
x1=f0['x1f']
x2=f0['x2f']
n1=len(x1)
n2=len(x2)
print(n1,n2)
n3=0
xx1=np.zeros((n1-1,n2-1))
yy1=np.zeros((n1-1,n2-1))
x11=x1[0:n1-1]
x22=x2[0:n2-1]
x11=x11.reshape(n1-1,1)
x22=x22.reshape(1,n2-1)
    
xx1=x11*np.sin(x22)
yy1=x11*np.cos(x22)

for k in range(k1,k2+1):
    filename='/perseus/scratch/gpfs/sashaph/Mach9highresstampede/BL.out2.%5.5d.athdf'%k
    outputname='Mach8.'
    
    #filename='/tigress/sashaph/restarttest/Mach9highres/BL.out2.00118.athdf'
    #print(filename)
    array=athdf(filename)

    data0=array['dens']
    
    data1=array['mom1']
 
    nonrot=data1[n3,:,:]/data0[n3,:,:]
    nonrot=nonrot.transpose()

    fig = plt.figure(figsize=(10,10))

    ax01 = plt.subplot(111)
    #,aspect='equal'
    plt.title(r"$\sqrt{\rho}V_r$")
    plt.xlabel(r"$x$",fontsize=18)
    plt.ylabel(r"$y$",fontsize=18)
    #im1=ax01.pcolormesh(xx1,yy1,nonrot,vmin=-0.01,vmax=0.01)
    #plt.colorbar(im1)
    plt.plot(nonrot[500,:])
    print(x1[500])
    #ax02 = plt.subplot(152,aspect='equal')
    #plt.title(r"Sigma")
    #plt.xlabel(r"$x$",fontsize=18)
    #plt.ylabel(r"$y$",fontsize=18)
    #im2=ax02.pcolormesh(xx1,yy1,psi1,vmin=0.,vmax=2.)
    #plt.colorbar(im2)
    
    outputname+=str(k)
    outputname+='.png'
    #plt.savefig(outputname) 
    #plt.close(fig)
    print(k)
    #print(xx1.shape)
    plt.show()
    #xx1.remove()
#,yy1[:,:],x22[:],x11[:],data1[:,:,:],x1[:],x2[:],data0[:,:,:],nonrot[:,:]
    gc.collect()    

