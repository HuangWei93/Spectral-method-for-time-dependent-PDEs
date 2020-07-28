import numpy as np
from numpy import *
from pylab import *
import sys
from utils import *


## Part B
#Collocation method
Ns = [16, 32, 48, 64, 96, 128, 192, 256]
step = 0.01
start= step
num = 100
cfls=np.arange(0,num)*step+start
errors = np.zeros((len(Ns), len(cfls)))
c = 4.0
nu = 0.1
for i, N in enumerate(Ns):
    h = 2*pi/(N+1); x = h*arange(0,N+1);
    u0 = exact_solution(x=x, t=0, nu=nu, c=c, tol=1e-18)
    #quadrature formula
    F = fourier(N)
    a0 = F.dot(u0)/(N+1)
    for j, cfl in enumerate(cfls):
        app_u = FourierCollocation_RK(u0=u0, x = x, t=np.pi/8.0, nu=nu, c = c, h=h, cfl=cfl, N=N)
        exact_u = exact_solution(x=x, t=np.pi/8.0, nu=nu, c=c, tol=1e-18)
        err = linalg.norm(app_u - exact_u,inf)
        errors[i,j] = err
fig = plt.figure()
fig.set_size_inches(18, 18)

for i, N in enumerate(Ns):
    plt.plot(cfls,errors[i,:], '-o', markersize=2, label = N)
    plt.yscale('log')
plt.legend()
plt.ylabel('$L^\infty$')
plt.xlabel('CFL')

plt.savefig('ex-b-collocation.png')


#Galerkin method
Ns = [16, 32, 48, 64, 96, 128, 192, 256]
step = 0.01
start= step
num = 300
cfls=np.arange(0,num)*step+start
errors = np.zeros((len(Ns), len(cfls)))
c = 4.0
nu = 0.1
for i, N in enumerate(Ns):
    h = 2*pi/(N+1); x = h*arange(0,N+1);
    u0 = exact_solution(x=x, t=0, nu=nu, c=c, tol=1e-16)
    #quadrature formula
    F = fourier(N)
    a0 = F.dot(u0)/(N+1)
    for j, cfl in enumerate(cfls):
        app_a = FourierGalerkin_RK(a0 = a0, x = x, t = np.pi/8.0, nu=nu, c = c, cfl=cfl, N=N)
        app_u = (F.conjugate()).T.dot(app_a)
        app_u = app_u.real
        exact_u = exact_solution(x=x, t=np.pi/8.0, nu=nu, c=c, tol=1e-16)
        err = linalg.norm(app_u - exact_u,inf)
        errors[i,j] = err

fig = plt.figure()
fig.set_size_inches(18, 18)

for i, N in enumerate(Ns):
    plt.plot(cfls,errors[i,:], '-o', markersize=2, label = N)
    plt.yscale('log')
plt.legend()
plt.ylabel('$L^\infty$')
plt.xlabel('CFL')

plt.savefig('ex-b-galerkin.png')

