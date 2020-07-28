import numpy as np
from numpy import *
from pylab import *
import sys
from utils import *


## Part C
Ns = [16, 32, 48, 64, 96, 128, 192, 256]
modes = ['collocation', 'galerkin']
errors = np.zeros((len(Ns), len(modes)))
c = 4.0
nu = 0.1
cfl = 0.05
for i, N in enumerate(Ns):
    h = 2*pi/(N+1); x = h*arange(0,N+1);
    u0 = exact_solution(x=x, t=0, nu=nu, c=c, tol=1e-16)
    #quadrature formula
    F = fourier(N)
    a0 = F.dot(u0)/(N+1)
    for j, mode in enumerate(modes):
        if mode == 'collocation':
            app_u = FourierCollocation_RK(u0=u0, x = x, t=np.pi/4.0, nu=nu, c = c, h=h, cfl=cfl, N=N)
            #app_u = FourierCollocation_RK(u0=u0, t=np.pi/4.0, num_ts=10000, N=N)
        else:
            app_a = FourierGalerkin_RK(a0 = a0, x = x, t = np.pi/4.0, nu=nu, c = c, cfl=cfl, N=N)
            #app_a = FourierGalerkin_RK(a0, t=np.pi/4.0, num_ts=10000)
            app_u = (F.conjugate()).T.dot(app_a)
        exact_u = exact_solution(x=x, t=np.pi/4.0, nu=nu, c=c, tol=1e-16)
        err = linalg.norm(app_u - exact_u,inf)
        errors[i,j] = err

fig, axs =plt.subplots(2,1)
clust_data = errors
rows = ['%d' % x for x in Ns]
collabel = modes
axs[0].axis('tight')
axs[0].axis('off')
the_table = axs[0].table(cellText=clust_data, rowLabels=rows, colLabels=collabel,loc='center')

axs[1].loglog(Ns,clust_data[:,0], '-o', label = modes[0])
axs[1].loglog(Ns,clust_data[:,1], '--^', label = modes[1])
plt.legend()
plt.ylabel('$L^\infty$')
plt.xlabel('N')
plt.savefig('ex-c.png')

## Part D
N = 128
h = 2*pi/(N+1); x = h*arange(0,N+1);
u0 = exact_solution(x=x, t=0, nu=nu, c=c, tol=1e-16)
F = fourier(N)
a0 = F.dot(u0)/(N+1)
app_us = []
ts = [0, np.pi/8.0, np.pi/6.0, np.pi/4.0]
t_names = ['t=0', 't=$\pi$/8', 't=$\pi$/6', 't=$\pi$/4']
#Visualization
plt.figure()
fig, axs = plt.subplots(3, 1)
fig.set_size_inches(9, 18)
modes = ['exact', 'collocation', 'galerkin']
for j, mode in enumerate(modes):
    lower = 1e10
    upper = 1e-10
    for t, t_name in zip(ts, t_names):
        if mode == 'collocation':
            ax = axs[1]
            u = FourierCollocation_RK(u0=u0, x = x, t=t, nu=nu, c = c, h=h, cfl=cfl, N=N)
            #u = FourierCollocation_RK(u0=u0, t=t, num_ts=10000, N=N)
            u = u.real
        elif mode == 'galerkin':
            ax = axs[2]
            app_a = FourierGalerkin_RK(a0 = a0, x=x, t = t, nu=nu, c =c, cfl=cfl, N=N)
            #app_a = FourierGalerkin_RK(a0, t=t, num_ts=10000)
            u = (F.conjugate()).T.dot(app_a)
            u = u.real
        else:
            ax = axs[0]
            u = exact_solution(x=x, t=t, nu=nu, c=c, tol=1e-16)
            u = u.real
        ax.plot(x,u,'.-',markersize=4, label = t_name)
        lower = np.minimum(np.min(u)- 0.3*abs(np.min(u)), lower)
        upper = np.maximum(np.max(u)+0.3*abs(np.max(u)), upper)
    ax.set_xlim(0,2*pi); ax.set_ylim(lower,upper);
    ax.set_title(mode)
    ax.legend(loc='upper left')
plt.savefig('ex-d.png')
