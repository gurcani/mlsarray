#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:24:19 2024

@author: ogurcan
"""
from time import time
import numpy as np

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
        else:
            lptr=grp[l]
            if(ext_flag):
                lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
                lptr[-1,]=m
            else:
                lptr[...]=m
            lptr.flush()
        fl.flush()

class gensolver:    
    def __init__(self,solver,f,t0,y0,t1,fsave,fshow=None,fy=None,dtstep=0.1,dtshow=None,dtsave=None,dtfupdate=None,force_update=None,dense=False,**kwargs):

        svs=solver.split(".")
        print(kwargs)        
        if(dtshow is None):
            dtshow=dtstep
        if(dtsave is None):
            dtsave=dtstep
        if isinstance(dtsave,float):
            dtsave=np.array([dtsave,])
        if isinstance(dtsave,list) or isinstance(dtsave,tuple):
            dtsave=np.array(dtsave)
        if (fshow is None):
            def fshow(t,u):
                print('t=',t,time()-self.ct,'secs elapsed, u^2 =',((u*u.conj()).real**2).sum())

        if svs[0]=='scipy':
            import scipy.integrate as scpint
            svf=getattr(scpint,svs[1])
            r=svf(f,t0,y0,t1,max_step=dtstep,**kwargs)
        if svs[0]=='cupy_ivp':
            from . import cupy_ivp as cpi
            svf=getattr(cpi,svs[1])
            r=svf(f,t0,y0,t1,max_step=dtstep,**kwargs)
        if not hasattr(r, 'integrate'):
            def integrate(tnext):
                while(r.t<tnext):
                    r.step()
            r.integrate=integrate
        self.r=r

        self.dtstep,self.dtshow,self.dtsave=dtstep,dtshow,dtsave
        self.t0,self.t1=t0,t1
        self.dense=dense
        if(not(fy is None) and not(force_update is None)):
            self.fy=fy
            self.force_update=force_update
            if(dtfupdate is None):
                dtfupdate=dtstep
            self.dtfupdate=dtfupdate
        if(callable(fsave)):
            self.fsave=[fsave,]
        else:
            self.fsave=fsave
        self.fshow=fshow
    
    def run(self):
        dtstep,dtshow,dtsave=self.dtstep,self.dtshow,self.dtsave
        dtfupdate=None
        t0,t1=self.t0,self.t1
        r=self.r
        #trnd=int(-np.log10(min(dtstep,dtshow,min(dtsave))/100))
        self.ct=time()
        t=t0
        tnext=t0+dtstep
        tshownext=t0+dtshow
        tsavenext=np.array([t0+dt for dt in dtsave])
        tifd = lambda t,td : td if(hasattr(self,'dense') and self.dense) else t

        if('dtfupdate' in self.__dict__.keys()):
            dtfupdate=self.dtfupdate
            tnextfupdate=t0+dtfupdate
        Nrnd=int(-np.log10(min(dtstep,dtshow,min(dtsave))/100))
        while(t<t1):
            r.integrate(tnext)
            t=r.t.item()
            tnext=tnext+dtstep
            if(not(dtfupdate is None)):
                if(t>=tnextfupdate):
                    tnextfupdate=tnextfupdate+dtfupdate
                    self.force_update(self.fy,t)
            if(t>=tshownext):
                print('t='+str(round(tifd(t,tshownext),Nrnd))+', '+str(time()-self.ct)+" secs elapsed." , end='')
                if(callable(self.fshow)):
                    self.fshow(round(tifd(t,tshownext),Nrnd),r.y)
                tshownext=tshownext+dtshow
            for l in range(len(dtsave)):
                if(t>=tsavenext[l]):
                    if(hasattr(self,'dense') and self.dense):
                        self.fsave[l](tsavenext[l],r.dense_output()(tsavenext[l]))
                    else:
                        self.fsave[l](t,r.y)
                    tsavenext[l]=tsavenext[l]+dtsave[l]
