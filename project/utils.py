import numpy as np
from numpy import *
from pylab import *
import sys

#Define the first order derivative differentiation matrix by odd and even methods
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
#Define Fourier coefficiences matrix
def fourier(N):
    F = np.zeros((N+1,N+1), dtype=complex)
    for  i in np.arange(-(N//2),N//2+1):
        for j  in range(N+1):
            F[i+N//2,j] = np.exp(-1j*i*j*2.0*np.pi/(N+1))
    return F
#Define Fourier Collocation with Runge-Kutta method
def FourierCollocation_RK(u0, x, t, nu, c, h, cfl, N):
    # Iterate for number of iterations
    u = u0
    if t == 0:
        return u
    D = odd_method(N)
    def OpF(u):
        nu = 0.1
        return -u*D.dot(u) + nu*D.dot(D.dot(u))
    update_t = 0
    while update_t < t:
        u_exact = exact_solution(x=x, t=update_t, nu=nu, c=c, tol=1e-18)
        delta_t = cfl/np.max(abs(u_exact)/h+nu/h**2)
        delta_t = np.minimum(t-update_t, delta_t)
        "Apply Runge Kutta Formulas to find next value of u"
        u1 = u + delta_t/2.0*OpF(u)
        u2 = u + delta_t/2.0*OpF(u1)
        u3 = u + delta_t*OpF(u2)
        u = 1.0/3.0*(-u + u1 + 2.0*u2 + u3 + delta_t/2.0*OpF(u3))
        update_t = update_t + delta_t
    return u

#Define Fourier Galerkin with Runge-Kutta method
def FourierGalerkin_RK(a0, x, t, nu, c, cfl, N):
    # Iterate for number of iterations
    a = a0
    if t == 0:
        return a
    length = len(a)
    def OpF(a):
        nu = 0.1
        vec = np.arange(-(length//2),length//2+1,dtype=np.float)
        return -1.0*np.convolve(1j*vec*a, a, 'same') - nu*vec**2*a
    update_t = 0
    F = fourier(N)
    while update_t < t:
        #u = (F.conjugate()).T.dot(a)
        u_exact = exact_solution(x=x, t=update_t, nu=nu, c=c, tol=1e-18)
        delta_t = cfl/np.max(abs(u_exact)*N/2.0+nu*(N/2.0)**2)
        delta_t = np.minimum(t-update_t, delta_t)
        "Apply Runge Kutta Formulas to find next value of u"
        a1 = a + delta_t/2.0*OpF(a)
        a2 = a + delta_t/2.0*OpF(a1)
        a3 = a + delta_t*OpF(a2)
        a = 1.0/3.0*(-a + a1 + 2.0*a2 + a3 + delta_t/2.0*OpF(a3))
        update_t = update_t + delta_t
    return a

#Define exact solution to Burger's equation
def phi(a, b, nu, tol):
    k = 0
    residual = exp(-(a-(2.0*k+1)*np.pi)**2/(4.0*nu*b))
    result = 0
    while (abs(residual) > tol).any():
        result = result + residual
        k = k + 1
        residual = exp(-(a-(2.0*k+1)*np.pi)**2/(4.0*nu*b)) + exp(-(a-(-2.0*k+1)*np.pi)**2/(4.0*nu*b))
    return result

def dphi(a, b, nu, tol):
    k = 0
    residual = exp(-(a-(2.0*k+1)*np.pi)**2/(4.0*nu*b)) * (-(a-(2*k+1)*np.pi)/(2*nu*b))
    result = 0
    while (abs(residual) > tol).any():
        result = result + residual
        k = k + 1
        residual = exp(-(a-(2.0*k+1)*np.pi)**2/(4.0*nu*b)) * (-(a-(2*k+1)*np.pi)/(2*nu*b)) + exp(-(a-(-2.0*k+1)*np.pi)**2/(4.0*nu*b)) * (-(a-(-2*k+1)*np.pi)/(2*nu*b))
    return result

def exact_solution(x, t, nu, c, tol):
    return c - 2.0*nu*dphi(x-c*t, t+1, nu, tol)/phi(x-c*t, t+1, nu, tol)

#Define  curvature of curves
def curvature(x, y):
#first derivatives
    dx= np.gradient(x)
    dy = np.gradient(y)

    #second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    #calculation of curvature from the typical formula
    cur = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
    return cur
