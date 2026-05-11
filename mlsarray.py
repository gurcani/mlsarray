#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:28:38 2024

@author: ogurcan
"""
import os
import sys
import numpy as np

get = lambda x : x.get() if (('cupy' in str(type(x))) or ('cupy' in str(x.__class__.__base__)))  else x

match os.environ.get('MLSARRAY_BACKEND'):
    case "cupy":
        import cupy as xp
        from cupyx.scipy.fft import rfft2,irfft2
    case "numpy":
        xp=np
        from scipy.fft import rfft2,irfft2
    case _:
        print("unknown backend: using numpy")
        xp=np
        from scipy.fft import rfft2,irfft2

class slicelist:
    def __init__(self,Nx,Ny):
        shp=(Nx,Ny)
        pshp=(int(np.ceil((Nx*3/2)/2)*2),int(np.ceil((Ny*3/2)/2)*2))
        insl=[np.s_[0:1,1:int(Ny/2)],np.s_[1:int(Nx/2),:int(Ny/2)],np.s_[-int(Nx/2)+1:,1:int(Ny/2)]]
        shps=[[len(range(*(l[j].indices(shp[j])))) for j in range(len(l))] for l in insl]
        Ns=[np.prod(l) for l in shps]
        outsl=[np.s_[sum(Ns[:l]):sum(Ns[:l])+Ns[l]] for l in range(len(Ns))]
        self.insl,self.shape,self.shps,self.Ns,self.outsl,self.pshp=insl,shp,shps,Ns,outsl,pshp

class mlsarray(xp.ndarray):
    def __new__(cls,Nx,Ny):
        v=xp.zeros((Nx,int(Ny/2)+1),dtype=complex).view(cls)
        return v
    def __getitem__(self,key):
        if(isinstance(key,slicelist)):
            return [xp.ndarray.__getitem__(self,l).ravel() for l in key.insl]
        else:
            return xp.ndarray.__getitem__(self,key)
    def __setitem__(self,key,value):
        if(isinstance(key,slicelist)):
            for l,j,shp in zip(key.insl,key.outsl,key.shps):
                self[l]=value.ravel()[j].reshape(shp)
        else:
            xp.ndarray.__setitem__(self,key,value)
        
def init_kspace_grid(sl):
    Nx,Ny=sl.shape
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0]
    kyl=np.r_[0:int(Ny/2+1)]
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    kx=xp.hstack([kx[l].ravel() for l in sl.insl])
    ky=xp.hstack([ky[l].ravel() for l in sl.insl])
    return kx,ky

def irft2(uk,sl):
    u=mlsarray(*sl.pshp)
    u[sl]=uk
    u[-1:-int(sl.shape[0]/2):-1,0]=u[1:int(sl.shape[0]/2),0].conj()
    u.view(dtype=float)[:,:-2]=irfft2(u,norm='forward',overwrite_x=True)
    return u.view(dtype=float)[:,:-2]

def rft2(u,sl):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return np.hstack(uk[sl])
