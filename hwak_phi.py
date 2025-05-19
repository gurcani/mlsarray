#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:19:37 2024

@author: ogurcan
"""

import cupy as xp
import numpy as np
from mlsarray import mlsarray,slicelist,init_kspace_grid,rfft2
from gensolver import gensolver
import h5py as h5
import os

tilde = lambda x : (x-xp.mean(x,axis=-1))

Npx,Npy=2048,2048
flname='out.h5'

t0,t1=0,10
dtstep,dtshow,dtsave=0.01,0.01,0.1
wecontinue=False
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=32*np.pi,32*np.pi
dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
w=10.0
phik=1e-4*xp.exp(-lkx**2/2/w**2-lky**2/w**2)*xp.exp(1j*2*np.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
nk=1e-4*xp.exp(-lkx**2/w**2-lky**2/w**2)*xp.exp(1j*2*np.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
zk=np.hstack((phik,nk))
xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=xp.meshgrid(xp.array(xl),xp.array(yl),indexing='ij')
kap0=1.0
C0=5.0
C1=0.5
nu0=5e-2
nu1=0.0
D0=5e-2
D1=0.0
mu1=1e2
mu0=0.0
kap_x=kap0-0.0*(x-Lx/2)**2
sig=Lx/100
nu_x=nu0+nu1*(xp.exp(-(x-x[0])**2/sig**2/2)+xp.exp(-(x-x[-1])**2/sig**2/2))
D_x=D0+D1*(xp.exp(-(x-x[0])**2/sig**2/2)+xp.exp(-(x-x[-1])**2/sig**2/2))
pen_x=(xp.exp(-(x-x[0])**4/sig**2/2)+xp.exp(-(x-x[-1])**4/sig**2/2))
C_x=(C0-C1)*(1-np.tanh((x-Lx/2)/1))/2+C1
u=mlsarray(Npx,Npy)
def irft(uk):
#    u.fill(0)
    u=mlsarray(Npx,Npy)
    u[sl]=uk
    u.irfft2()
    return u.view(dtype=float)[:,:-2]

def rft(u):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return xp.hstack(uk[sl])

def save_data(fl,grpname,ext_flag,**kwargs):
    if not (grpname in fl):
        grp=fl.create_group(grpname)
    else:
        grp=fl[grpname]
    for l,m in kwargs.items():
        if not l in grp:
            if(not ext_flag):
                grp[l]=m
            else:
                if(np.isscalar(m)):
                    grp.create_dataset(l,(1,),maxshape=(None,),dtype=type(m))
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                else:
                    grp.create_dataset(l,(1,)+m.shape,chunks=(1,)+m.shape,maxshape=(None,)+m.shape,dtype=m.dtype)
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                lptr=grp[l]
                lptr[-1,]=m
                lptr.flush()
        elif(ext_flag):
            lptr=grp[l]
            lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
            lptr[-1,]=m
            lptr.flush()
        fl.flush()

def save_callback(fl,t,y):
    zk=y.view(dtype=complex)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    om=irft(-phik*(kx**2+ky**2))
    n=irft(nk)
    save_data(fl,'fields',ext_flag=True,om=om.get(),n=n.get(),t=t)

def rhs(t,y):
    zk=y.view(dtype=complex)
    dzkdt=xp.zeros_like(zk)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    dphikdt,dnkdt=dzkdt[:int(zk.size/2)],dzkdt[int(zk.size/2):]
    ksqr=kx**2+ky**2
    dxphi=irft(1j*kx*phik)
    dyphi=irft(1j*ky*phik)
    om=irft(-ksqr*phik)
    n=irft(nk)
    phi=irft(phik)
    d2n=irft(-ksqr*nk)
    deltan_ksqrinv=irft((phik-nk)/ksqr)
    dphikdt[:]=-1j*kx*rft(dyphi*om)/ksqr+1j*ky*rft(dxphi*om)/ksqr
    dnkdt[:]=1j*kx*rft(dyphi*n)-1j*ky*rft(dxphi*n)
    dphikdt[:]+=rft((-C_x*tilde(deltan_ksqrinv))*(1-pen_x)+nu_x*om-mu1*pen_x*phi)
    dnkdt[:]+=rft((-kap_x*dyphi+C_x*tilde(phi-n))*(1-pen_x)+D_x*d2n-mu1*pen_x*n)
    return dzkdt.view(dtype=float)

if(wecontinue):
    fl=h5.File('out2.h5','r+',libver='latest')
    fl.swmr_mode = True
    omk,nk=rft(xp.array(fl['fields/om'][-1,])),rft(xp.array(fl['fields/n'][-1,]))
    phik=-omk/(kx**2+ky**2)
    t0=fl['fields/t'][-1]
    zk=np.hstack((phik,nk))
else:
    if os.path.exists(flname):
        os.remove(flname)
    fl=h5.File(flname,'w',libver='latest')
    fl.swmr_mode = True
    save_data(fl,'data',ext_flag=False,x=x.get(),y=y.get(),kap_x=kap_x.get(),C_x=C_x.get(),nu_x=nu_x.get(),D_x=D_x.get())
fsave=lambda t,y : save_callback(fl,t,y)
r=gensolver('cupy_ivp.DOP853',rhs,t0,zk.view(dtype=float),t1,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,dense_output=True,rtol=1e-9,atol=1e-10)
r.run()
fl.close()
