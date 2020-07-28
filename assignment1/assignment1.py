import numpy as np
from numpy import *
from pylab import *
import sys

#Set parameters of function of which derivative we approximate
k = float(sys.argv[1])

N = 1
err = 1e12
tol = 1e-5
while not err <= tol:
    # Set up grid and differentiation matrix with sin:
    N = N + 1; h = 2*pi/(N+1); x = h*arange(0,N+1);
    D = np.zeros((N+1,N+1))
    for j in range(N+1):
        for i in range(N+1):
            if j!=i:
                D[j,i] = 0.5*(-1)**(j+i)*(sin((j-i)*np.pi/(N+1)))**(-1)
    v = exp(k*sin(x)); vprime = k*cos(x)*v; app_vprime = dot(D,v);
    err = linalg.norm(app_vprime-vprime,inf)

#Visualization
fig = plt.figure(figsize=(15,6))
fig.suptitle("k = {0}, N = {1}, max_error = {2}".format(int(k), int(N), err), fontsize=16)

#original function
ax = plt.subplot("131")
ax.plot(x,v,'.-',markersize=8)
ax.set_xlim(0,2*pi); ax.set_ylim(np.min(v)- 0.1*abs(np.min(v)),np.max(v)+0.1*abs(np.max(v)));
ax.set_title("original function")
#exact derivative function
ax = plt.subplot("132")
ax.plot(x,vprime,'.-',markersize=8)
ax.set_xlim(0,2*pi); ax.set_ylim(np.min(vprime)- 0.1*abs(np.min(vprime)),np.max(vprime)+0.1*abs(np.max(vprime)));
ax.set_title("exact derivative function")
#numerical derivative function
ax = plt.subplot("133")
ax.plot(x,app_vprime,'.-',markersize=8)
ax.set_xlim(0,2*pi); ax.set_ylim(np.min(vprime)- 0.1*abs(np.min(vprime)),np.max(vprime)+0.1*abs(np.max(vprime)));
ax.set_title("numerical derivative function")
show()
print("N = {0} max_error = {1}".format(int(N), err))
