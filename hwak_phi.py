#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:19:37 2024

@author: ogurcan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:02:31 2025

@author: ogurcan
"""

import numpy as np
import cupy as cp
from mlsarray_gpu import mlsarray,slicelist,init_kspace_grid,rfft2
from gensolver import gensolver,save_data
import h5py as h5
xp=cp

#Npx,Npy=2048,2048
Npx,Npy=2048,2048
t0,t1=0,400
dtstep,dtshow,dtsave=0.1,0.1,1.0
wecontinue=False
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=12*np.pi,12*np.pi
dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
w=10.0
phik=1e-4*xp.exp(-lkx**2/2/w**2-lky**2/w**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
nk=1e-4*xp.exp(-lkx**2/w**2-lky**2/w**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
zk0=xp.hstack((phik,nk))
xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')
kap=1.0
C=1.0
nu=1e-5
D=1e-5
u=mlsarray(Npx,Npy)
    # del om,n
    # gc.collect()
    # xp.get_default_memory_pool().free_all_blocks() 

def irft(uk):
    u=mlsarray(Npx,Npy)
    u[sl]=uk
    u.irfft2()
    return u.view(dtype=float)[:,:-2]

def rft(u):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return np.hstack(uk[sl])

def save_callback(fl,t,zk):
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    om=irft(-phik*(kx**2+ky**2))
    n=irft(nk)
    save_data(fl,'fields',ext_flag=True,om=om.get(),n=n.get(),t=t)

def rhs(dzkdt,zk,t):
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    dphikdt,dnkdt=dzkdt[:int(zk.size/2)],dzkdt[int(zk.size/2):]
    ksqr=kx**2+ky**2
    dxphi=irft(1j*kx*phik)
    dyphi=irft(1j*ky*phik)
    om=irft(-ksqr*phik)
    n=irft(nk)
    sigk=(ky>0)
    dphikdt[:]=(-1j*kx*rft(dyphi*om)+1j*ky*rft(dxphi*om)-C*(phik-nk)*sigk)/ksqr-nu*sigk*ksqr**2*phik
    dnkdt[:]=(1j*kx*rft(dyphi*n)-1j*ky*rft(dxphi*n)-1j*kap*ky*phik+C*(phik-nk)*sigk)-D*sigk*ksqr**2*nk

dzk=np.zeros_like(zk0)

if(wecontinue):
    fl=h5.File('out2.h5','r+',libver='latest')
    fl.swmr_mode = True
    omk,nk=rft(np.array(fl['fields/om'][-1,])),rft(np.array(fl['fields/n'][-1,]))
    phik=-omk/(kx**2+ky**2)
    t0=fl['fields/t'][-1]
    zk=xp.hstack((phik,nk))
else:
    fl=h5.File('out.h5','w',libver='latest')
    fl.swmr_mode = True
    save_data(fl,'data',ext_flag=False,x=x,y=y,kap=kap,C=C,nu=nu,D=D)

fsave = lambda t,y : save_callback(fl,t,y)

#r=gensolver('scipy.DOP853',rhs,t0,zk0,t1,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=1e-9,atol=1e-10)
#r=gensolver('julia.KenCarp4(autodiff=false,linsolve = KrylovJL_GMRES())',rhsexp,t0,zk0,t1,fimp=rhsimp,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,reltol=1e-9,abstol=1e-10)
#r=gensolver('julia.BS3()',rhs,t0,zk0,t1,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,reltol=1e-9,abstol=1e-10)
r=gensolver('julia.Tsit5()',rhs,t0,zk0,t1,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,reltol=1e-9,abstol=1e-10)
#r=gensolver('julia.lsoda()',rhs,t0,zk0,t1,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,reltol=1e-9,abstol=1e-10)
#r=gensolver('cupy_ivp.DOP853',rhs,t0,zk0,t1,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=1e-9,atol=1e-10)
r.run()
fl.close()
