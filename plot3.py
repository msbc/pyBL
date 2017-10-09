import h5py
from numpy import array
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import  matplotlib
from matplotlib import rc
from athena_read import athdf as athdf
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]

k1=int(raw_input('First:'))
k2=int(raw_input('Last:'))
for k in range(k1,k2+1):
    #filename='/perseus/scratch/gpfs/sashaph/M9MHDstampede/BL.out2.00'
    filename='/tigress/sashaph/restarttest/lowres/BL.out2.00'
    #filename='/tigress/cjwhite/mad/mad_smr_hlle.out1.00'
    if k<10:
        filename+='00'
        filename+=str(k)
        filename+='.athdf'

    if k>9 and k<100:
        filename+='0'
        filename+=str(k)
        filename+='.athdf'

    if k>99:
        filename+=str(k)
        filename+='.athdf'
    
    #filename='/tigress/sashaph/BL3D/BL.smr.example.athdf'

    array=athdf(filename,subsample=True)
    #print(array.keys())
    data0=array['dens']
    x01=array['x1f']
    x02=array['x2f']
    x03=array['x3f']
    data1=array['mom1']
    data2=array['mom2']
    data3=array['mom3']
    data4=array['Bcc1']
    data5=array['Bcc2']
    data6=array['Bcc3']
    n1=len(x01)
    n2=len(x02)
    n3=len(x03)
    nn=(n2-1)/2.
    #print(n1,n2,n3)
    xx1=x01[0:n1-1]
    xx2=x02[0:n2-1]
    xx3=x03[0:n3-1]
    
    xx1=xx1.reshape(n1-1,1)
    xx3=xx3.reshape(1,n3-1)
    x1=xx1*np.sin(xx3)
    y1=xx1*np.cos(xx3)
   
    
    yy1=x01[0:n1-1]
    yy2=x02[0:n2-1]
    yy3=x03[0:n3-1]

    yy1=yy1.reshape(n1-1,1)
    yy2=yy2.reshape(1,n2-1)
    x2=yy1*np.sin(yy2)
    y2=yy1*np.cos(yy2)
    
    
    psi1=np.zeros((n1-1,n3-1))
    psi1=data1[:,nn,:]/(data0[:,nn,:])**0.5
    #psi1=data6[:,nn,::]
    psi1=psi1.transpose()

    densav=np.mean(data0,0)
    psi3=densav[:,:]
    dt1=np.mean(psi3,0)
    #psi3=data0[30,:,:]
    #psi3=data0[50,:,:]
    psi3=psi3.transpose()
    
    
    psi2=np.mean(data6,0)
    psi2=psi2.transpose()
    

    psi5=np.mean(data3,0)
    psi5=psi5.transpose()
    psi5=psi5*x2

    psi6=np.mean(data2/data0**0.5,0)
    psi6=psi6.transpose()

    #beta=data4**2.+data5**2+data6**2./(data0*0.2**2)
    #betaav=np.mean(beta,0)
    #psi3=betaav[:,:]
    #psi3=psi3.transpose()

    xf=x01[0:n1-1]
    tt=data3[:,nn,:]/data0[:,nn,:]
    dt=np.mean(tt,axis=0)

    print(psi5.shape)

    fig = plt.figure(figsize=(5, 5))

    ax01 = plt.subplot(111)
    plt.plot(xf,psi5[:,nn])
    plt.xlim(0.751,2.5)
    plt.ylim(-0.2,20.)


#plt.plot(xf[:],np.log10(data1[100,128,:]))
#plt.plot(xf[:],data[100,128,:]/data1[100,128,:])
#plt.ylim(-.001,.001)
    outputname='moment'
    outputname+=str(k)
    outputname+='.png'
    plt.savefig(outputname) 
    plt.close(fig)
    print(k)
    #plt.show()
        

