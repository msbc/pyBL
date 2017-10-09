
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
    filename='/perseus/scratch/gpfs/sashaph/MHD9outflow/BL.out2.00'
    #filename='/tigress/sashaph/restarttest/lowres/BL.out2.00'
    #filename='/perseus/scratch/gpfs/sashaph/MHDpolar/BL.out2.00'
    #filename='/perseus/scratch/gpfs/sashaph/MHDM6smr/BL.out2.00'
    #filename='/perseus/scratch/gpfs/sashaph/MHDM9stampede/BL.out2.00'
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

    psi7=np.mean(data1[:,:,:],0)
    psi7=psi7.transpose()
    #print(psi7.shape,xx3.shape,xx1.shape)
    psi7=np.power(yy1,2)*np.sin(yy2)*psi7
    dt2=np.mean(psi7,axis=1)
    dt2=np.log10(np.abs(dt2+0.0000000000000001))

    fig = plt.figure(figsize=(18, 9))

    ax01 = plt.subplot(231,aspect='equal')
    plt.title(r"$\sqrt{\rho}V_r$")
    plt.xlabel(r"$x$",fontsize=18)
    plt.ylabel(r"$y$",fontsize=18)
    im1=ax01.pcolormesh(x1,y1,psi1,vmin=-0.05,vmax=0.05)
    plt.colorbar(im1)
    #plt.xlim(-1.5,1.5)
    #plt.ylim(-1.5,1.5)
    plt.xlim(0,1.5)
    plt.ylim(0,1.5)

    ax02 = plt.subplot(232)
    plt.title(r"Bphav")
    plt.xlabel(r"$x$",fontsize=18)
    plt.ylabel(r"$z$",fontsize=18)
    im2=ax02.pcolormesh(x2,y2,psi2,vmin=-0.03,vmax=0.03)
    plt.colorbar(im2)
    plt.xlim(0,4)
    plt.ylim(-2,2.)
    

    ax03 = plt.subplot(233)
    plt.title(r"densav")
    plt.xlabel(r"$x$",fontsize=18)
    plt.ylabel(r"$z$",fontsize=18)
    im3=ax03.pcolormesh(x2,y2,psi3,vmin=0.,vmax=1.5)
    plt.colorbar(im3)
    plt.xlim(0,2)
    plt.ylim(-1,1.)
    
    ax04 = plt.subplot(234)
    plt.title(r"$<V_{\phi}>$")
    plt.xlabel(r"$r$",fontsize=18)
    plt.plot(xf,dt,'ro')
    plt.ylim(-0.1,1.1)
    plt.xlim(0.8,2.)

    ax05 = plt.subplot(235)
    plt.title(r"$<densav>$")
    plt.xlabel(r"$r$",fontsize=18)
    plt.plot(xf,dt1,'ro')
    plt.ylim(-0.1,1.1)
    plt.xlim(0.8,3.)

    ax06 = plt.subplot(236)
    plt.title(r"$<\rho V_{\phi} R>$")
    plt.xlabel(r"$x$",fontsize=18)
    plt.ylabel(r"$z$",fontsize=18)
    im6=ax06.pcolormesh(x2,y2,psi5,vmin=0,vmax=3.)
    plt.colorbar(im6)
    plt.xlim(0.5,1.5)
    plt.ylim(-0.5,0.5)



#plt.plot(xf[:],np.log10(data1[100,128,:]))
#plt.plot(xf[:],data[100,128,:]/data1[100,128,:])
#plt.ylim(-.001,.001)
    #outputname='MHD6smr'
    #outputname='MHDpolar'
    outputname='MHD9outflow'
    #outputname='MHD9stamp'
    outputname+=str(k)
    outputname+='.png'
    plt.savefig(outputname) 
    plt.close(fig)
    print(k)
    #plt.show()
        

