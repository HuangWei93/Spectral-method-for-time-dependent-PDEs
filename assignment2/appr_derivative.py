import numpy as np
from numpy import *
from pylab import *
import sys


#Set parameters of function of which derivative we approximate
k = float(sys.argv[1])

def odd_method(N):
    D = np.zeros((N+1,N+1))
    for j in range(N+1):
        for i in range(N+1):
            if j!=i:
                D[j,i] = 0.5*(-1)**(j+i)*(sin((j-i)*np.pi/(N+1)))**(-1)
    return D

def even_method(N):
    D = np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            if j!=i:
                D[j,i] = 0.5*(-1)**(j+i) * (tan((j-i)*np.pi/N))**(-1)
    return D

N = 0
err = 1e12
tol = 1e-5
while not err <= tol:
    # Set up grid and differentiation matrix with sin:
    N = N + 2;
    if sys.argv[2] == 'o':
        D = odd_method(N)
        h = 2*pi/(N+1); x = h*arange(0,N+1);
    elif sys.argv[2] == 'e':
        D = even_method(N)
        h = 2*pi/N; x = h*arange(0,N);
    v = exp(k*sin(x)); vprime = k*cos(x)*v; app_vprime = dot(D,v);
    err = linalg.norm(app_vprime-vprime,inf)
if sys.argv[2] == 'o':
    msg = "odd method: N = {0} max_error = {1}".format(int(N), err)
else:
    msg = "even method: N = {0} max_error = {1}".format(int(N), err)

print(msg)
