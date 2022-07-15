#!/usr/bin/env python
# coding: utf-8

# In[1]:


from casadi import *
import matplotlib.pyplot as plt
from math import pi
import numpy as np
def shift(state,sysinput,T):
    state=state+T*f(state,sysinput)
    return state
    
T=0.2 
N=10
rod_d=0.3
V_max=0.6
V_min=-V_max 
omega_max=pi/4
omega_min=-omega_max 
#states 
x=SX.sym('x')
y=SX.sym('y')
theta=SX.sym('theta')
states=vertcat(x,y,theta)
n_state=states.size1()
#controls
v=SX.sym('v')
omega=SX.sym('omega')
controls=vertcat(v,omega)
n_cont=controls.size1()
rhs=vertcat(v*cos(theta),v*sin(theta),omega)
f=Function('f',[states,controls],[rhs])
X=SX.sym('X',n_state,N+1)
U=SX.sym('U',n_cont,N)
P=SX.sym('P',2*n_state)

X[:,0]=P[0:3]
for i in range(N):
    st=X[:,i]
    con=U[:,i]
    f_value=f(st, con)
    st_next=st+T*f_value
    X[:,i+1]=st_next
ef=Function('ef',[P,U],[X])
obj=0
g=[]
Q= diag(SX([1,5,0.1]))
R= diag(SX([0.5,0.05]))
for i in range(N):
    st=X[:,i]
    con=U[:,i]
    obj=obj+(st-P[3:6]).T@Q@(st-P[3:6])+con.T@R@con
g=horzcat(X[1,:],X[2,:])
g=reshape(g,2*(N+1),1)
U=reshape(U,2*N,1)
nlp={}

nlp['x']=U
nlp['f']=obj
nlp['g']=g
nlp['p']=P
F=nlpsol('F','ipopt',nlp);
t0=0
x_g=[0 for i in range(2*N)]
x_i=[0,0,0]
xs=[1.5,1.5,0]
pini=x_i+xs
low=[]
up=[]
for i in range(2*N):
    if i%2==0:
        low.append(-0.6)
        up.append(0.6)
    else:
        low.append(-pi/4)
        up.append(pi/4)
u0 = [[0,0,0],[0,0,0]]
sim_time=20
token=0
t=[]
x_i_arr=[]
x_a=[]
y_a=[]
for i in range(40):
    H=F(x0=x_g,lbx=low,ubx=up,lbg=-2,ubg=2,p=pini)
    x_g=[]
    for i in range(2*N-2):
        x_g.append(float(H['x'][2+i]))
    x_g.append(float(H['x'][2*N-2]))
    x_g.append(float(H['x'][2*N-1]))
    H['x']=reshape(H['x'],2,N)
    state_n=ef(pini,H['x'])
    x_0= shift(state_n[:,0],H['x'][:,0],T)
    x_i=[float(x_0[0]),float(x_0[1]),float(x_0[2])]
    x_a.append(x_i[0])
    y_a.append(x_i[1])
    pini=x_i+xs
    x_i_arr.append(x_i)
t=np.arange(0,8,0.2)
print(x_a)
print(y_a)
plt.plot(t,x_i_arr)
plt.grid()
plt.show





