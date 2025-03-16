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
    def __init__(self,solver,fexp,t0,y0,t1,fsave,fshow=None,fy=None,dtstep=0.1,dtshow=None,dtsave=None,dtfupdate=None,force_update=None,fimp=None,**kwargs):

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
        if hasattr(y0,'__module__'):
            if(y0.__module__=="cupy"):
                import cupy as cp
                CUPY_ARRAY=True
                self.cp=cp
            else:
                CUPY_ARRAY=False
        else:
            CUPY_ARRAY=False
        
        if svs[0]=='julia':
            from juliacall import Main as jl
            jl.seval("""
                using DifferentialEquations
                using PythonCall
                using DiffEqCallbacks
                """)
            if CUPY_ARRAY:
                jl.fexp = lambda dy,y,p,t : fexp(self.jltocp(dy,y0),self.jltocp(y,y0),t)
            else:
                jl.fexp = lambda dy,y,p,t : fexp(np.array(dy,copy=False),np.array(y,copy=False),t)
            if(callable(fimp)):
                if CUPY_ARRAY:
                    jl.fimp=lambda dy,y,p,t : fimp(self.jltocp(dy,y0),self.jltocp(y,y0),t)
                else:
                    jl.fimp=lambda dy,y,p,t : fimp(np.array(dy,copy=False),np.array(y,copy=False),t)
            if(callable(fsave)):
                if CUPY_ARRAY:
                    jl.fsave=[lambda r : fsave(r.t,self.jltocp(r.u,y0)),]
                else:
                    jl.fsave=[lambda r : fsave(r.t,np.array(r.u,copy=False)),]                
            else:
                if CUPY_ARRAY:
                    jl.fsave=[ lambda r : f(r.t,self.jltocp(r.u,y0)) for f in fsave]
                else:
                    jl.fsave=[ lambda r : f(r.t,np.array(r.u,copy=False)) for f in fsave]
            if CUPY_ARRAY:
                jl.fshow = lambda r : fshow(r.t,self.jltocp(r.u,y0))
            else:
                jl.fshow = lambda r : fshow(r.t,np.array(r.u,copy=False))
            jl.dtsave = dtsave
            jl.dtshow = dtshow
            if CUPY_ARRAY:
                jl.y0_ptr=y0.data.ptr
                jl.seval("""
                    using CUDA
                    y0_p=CuPtr{ComplexF64}(pyconvert(UInt, y0_ptr))
                    """)
                jl.y0 = jl.unsafe_wrap(jl.CuArray, jl.y0_p, y0.size)
                jl.dy=cp.zeros_like(y0)
                # dy0=cp.zeros_like(y0)
                # jl.dy0_ptr=dy0.data.ptr
#                    dy0_p=CuPtr{ComplexF64}(pyconvert(UInt, dy0_ptr))
#                jl.dy = jl.unsafe_wrap(jl.CuArray, jl.dy0_p, dy0.size)
            else:
                jl.y0=y0
                jl.dy=jl.y0.copy()
                jl.dy.fill(0)
            jl.tspan=(t0,t1)
            jl.py_kwargs=kwargs
            jl.svtype=jl.seval(svs[1])
            jl.seval("""
                cbshow = PeriodicCallback(fshow,dtshow)
                global cbs=CallbackSet(cbshow)
                for l in range(1,length(fsave))
                    global cbs=CallbackSet(cbs,PeriodicCallback(fsave[l],dtsave[l]))
                end
                tmpdict = Dict{String, Any}(py_kwargs)
                kwdict = Dict()
                for (k,v) in tmpdict
                    kwdict[Symbol(k)] = v
                end
                """)
            if(callable(fimp)):
                    # using LinearAlgebra
                    # jp=Diagonal(y0)
                    # fimpode=ODEFunction(fimp;jac_prototype=jp)
                jl.seval("""
                    prob=SplitODEProblem(fimp,fexp,y0,tspan)
                """)
            else:
                jl.seval("""
                    prob=ODEProblem{true}(fexp,y0,tspan)
                """)
            self.jl=jl
            self.run=self.run_julia
        else:
            if svs[0]=='scipy':
                import scipy.integrate as scpint
                svf=getattr(scpint,svs[1])
                r=svf(fexp,t0,y0,t1,max_step=dtstep,**kwargs)
            if svs[0]=='cupy_ivp':
                import cupy_ivp as cpi
                svf=getattr(cpi,svs[1])
                if CUPY_ARRAY:
                    xp=cp
                else:
                    xp=np
                def f(t,y):
                    dy=cp.zeros_like(y0)
                    fexp(dy,xp.array(y,copy=False),t)
                    return dy
                r=svf(f,t0,y0,t1,max_step=dtstep,**kwargs)
            def integr(ti):
                while(r.t<ti):
                    r.step()
            r.integrate=integr
            self.r=r
            self.run=self.run_byhand
            if(callable(fsave)):
                self.fsave=[lambda t,y : fsave(t.get() if CUPY_ARRAY else t,y) ,]
            else:
                self.fsave=[lambda t,y : fl(t.get() if CUPY_ARRAY else t,y) for fl in fsave]
        self.dtstep,self.dtshow,self.dtsave=dtstep,dtshow,dtsave
        self.t0,self.t1=t0,t1
        if(not(fy is None) and not(force_update is None)):
            self.fy=fy
            self.force_update=force_update
            if(dtfupdate is None):
                dtfupdate=dtstep
            self.dtfupdate=dtfupdate
        self.fshow=fshow
        self.solver=solver
        
    def jltocp(self,y,zk):
        p=self.jl.UInt64(self.jl.pointer(y))
        mem = self.cp.cuda.UnownedMemory(p,zk.nbytes, None)
        memptr = self.cp.cuda.MemoryPointer(mem, 0)
        vk=self.cp.ndarray(zk.shape,dtype=zk.dtype,memptr=memptr)
#        print(hex(p),':',hex(vk.data.ptr))
        
        return vk

    def run_julia(self):
        self.ct=time()
        self.jl.seval("""
           sol = solve(prob,svtype,save_on=false,save_everystep=false,save_start=false,save_end=false,dense=false,callback=cbs;kwdict...)
        """)
        self.sol=self.jl.sol

    def run_byhand(self):
        self.ct=time()
        dtstep,dtshow,dtsave=self.dtstep,self.dtshow,self.dtsave
        dtfupdate=None
        t0,t1=self.t0,self.t1
        r=self.r
        trnd=int(-np.log10(min(dtstep,dtshow,min(dtsave))/100))
        t=t0
        tnext=round(t0+dtstep,trnd)
        tshownext=round(t0+dtshow,trnd)
        tsavenext=np.array([round(t0+l,trnd) for l in dtsave])
        if('dtfupdate' in self.__dict__.keys()):
            dtfupdate=self.dtfupdate
            tnextfupdate=round(t0+dtfupdate,trnd)        
        while(t<t1):
            r.integrate(tnext)
            tnext=round(tnext+dtstep,trnd)
            t=r.t
            if(not(dtfupdate is None)):
                if(r.t>=tnextfupdate):
                    tnextfupdate=round(tnextfupdate+dtfupdate,trnd)
                    self.force_update(self.fy,t)
            if(r.t>=tshownext):
#                print('t='+str(t)+', '+str(time()-self.ct)+" secs elapsed." , end='')
                if(callable(self.fshow)):
                    self.fshow(r.t,r.y)
                else:
                    print()
                tshownext=round(tshownext+dtshow,trnd)
            for l in range(len(dtsave)):
                if(r.t>=tsavenext[l]):
                    tsavenext[l]=round(tsavenext[l]+dtsave[l],trnd)
                    self.fsave[l](r.t,r.y)

    