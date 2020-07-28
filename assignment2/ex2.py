import numpy as np
from numpy import *
from pylab import *
import sys
import matplotlib.pyplot as plt
# Python program to implement Runge Kutta method
def odd_method(N):
    D = np.zeros((N+1,N+1))
    for j in range(N+1):
        for i in range(N+1):
            if j!=i:
                D[j,i] = 0.5*(-1)**(j+i)*(sin((j-i)*np.pi/(N+1)))**(-1)
    return D

def second_method(N):
    D = np.zeros((N+1,N+1))
    h = 2.0*np.pi/(N+1)
    for i in range(N+1):
        D[i, (i-1)%(N+1)] = -1.0/(2.0*h)
        D[i, (i+1)%(N+1)] = 1.0/(2.0*h)
    return D

def forth_method(N):
    D = np.zeros((N+1,N+1))
    h = 2.0*np.pi/(N+1)
    for i in range(N+1):
        D[i, (i-2)%(N+1)] = 1.0/(12.0*h)
        D[i, (i-1)%(N+1)] = -8.0/(12.0*h)
        D[i, (i+1)%(N+1)] = 8.0/(12.0*h)
        D[i, (i+2)%(N+1)] = -1.0/(12.0*h)
    return D

# Finds value of u using N
# and initial value u0.
def rungeKutta(u0, t, num_ts, N, mode):
    # Iterate for number of iterations 
    u = u0
    if t == 0:
        return u
    delta_t = float(t)/float(num_ts)
    if mode == '2nd':
        D = second_method(N)
    elif mode == '4th':
        D = forth_method(N)
    elif mode == 'inf':
        D = odd_method(N)
    for i in range(num_ts): 
        "Apply Runge Kutta Formulas to find next value of u"
        du = D.dot(u)
        u1 = u - 2.0*np.pi*delta_t/2.0*du
        du = D.dot(u1)
        u2 = u - 2.0*np.pi*delta_t/2.0*du
        du = D.dot(u2)
        u3 = u - 2.0*np.pi*delta_t*du
        du = D.dot(u3)
        u = 1.0/3.0*(-u + u1 + 2.0*u2 + u3 - 2.0*np.pi*delta_t/2.0*du)
    return u

#========part a======================
Ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
modes = ['2nd', '4th', 'inf']
errors = np.zeros((len(Ns), len(modes)))
for i, N in enumerate(Ns):
    h = 2*pi/(N+1); x = h*arange(0,N+1);
    u0 = exp(sin(x))
    for j, mode in enumerate(modes):
        app_u = rungeKutta(u0, np.pi, 10000, N, mode)
        exact_u = exp(sin(x - 2.0*np.pi*np.pi))
        err = linalg.norm(app_u - exact_u,inf)
        errors[i,j] = err

fig, axs =plt.subplots(2,1)
clust_data = errors
rows = ['%d' % x for x in Ns]
collabel=("2nd", "4th", "inf")
axs[0].axis('tight')
axs[0].axis('off')
the_table = axs[0].table(cellText=clust_data, rowLabels=rows, colLabels=collabel,loc='center')

axs[1].loglog(Ns,clust_data[:,0], '-o', label = modes[0])
axs[1].loglog(Ns,clust_data[:,1], '--v', label = modes[1])
axs[1].loglog(Ns,clust_data[:,2], '-.^', label = modes[2])
plt.legend()
plt.ylabel('$L^\infty$')
plt.xlabel('N')
plt.savefig('ex2-a.png')

#=======part b=========
plt.figure()
fig, axs = plt.subplots(3, 2)
fig.set_size_inches(12, 18)
ts = [0, 100, 200]
error_2nd = []
error_inf = []
for i, t in enumerate(ts):
    N1 = 200; N2 = 10;
    h1 = 2*pi/(N1+1); x1 = h1*arange(0,N1+1);
    u0 = exp(sin(x1))
    app_u1 = rungeKutta(u0, t, round(10000*(t/np.pi)), N1, '2nd')
    exact_u1 = exp(sin(x1 - 2.0*np.pi*t))
    err1 = linalg.norm(app_u1 - exact_u1,inf)
    h2 = 2*pi/(N2+1); x2 = h2*arange(0,N2+1);
    u0 = exp(sin(x2))
    app_u2 = rungeKutta(u0, t, round(10000*(t/np.pi)), N2, 'inf')
    exact_u2 = exp(sin(x2 - 2.0*np.pi*t))
    err2 = linalg.norm(app_u2 - exact_u2,inf)
    error_2nd.append(err1)
    error_inf.append(err2)
    axs[i,0].plot(x1, exact_u1, '--', label = 'exact')
    axs[i,0].plot(x1, app_u1, '-', label = '2nd')
    axs[i,0].set_title('2nd method N = 200')
    axs[i,0].set_xlabel('x')
    axs[i,0].set_ylabel('$u(x,{})$'.format(t))
    axs[i,0].legend()
    
    axs[i,1].plot(x2, exact_u2, '--', label = 'exact')
    axs[i,1].plot(x2, app_u2, '-', label = 'spectral')
    axs[i,1].set_title('spectral method N = 10')
    axs[i,1].set_xlabel('x')
    axs[i,1].set_ylabel('$u(x,{})$'.format(t))
    axs[i,1].legend()

plt.savefig('ex2-b1.png')

plt.figure()
plt.plot(ts, error_2nd, '-o', label = '2nd')
plt.plot(ts, error_inf, '--v', label = 'inf')
plt.legend()
plt.ylabel('$L^\infty$')
plt.xlabel('t')
plt.savefig('ex2-b2.png')
