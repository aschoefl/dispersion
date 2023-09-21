from ngsolve import *
import scipy
import numpy as np

def TOAR (A, B, maxiter=200, start=None):

    # declare reused variables
    r = A.CreateVector()
    w = A.CreateVector()
    Q = MultiVector(r,0)
    T = Matrix(maxiter, maxiter-1, complex=A.is_complex)

    # define q0
    if start:
        r.data = start
    else:
        r.SetRandom()
        r /= sqrt(InnerProduct(r,r).real)

    # assign rank
    eta = 1

    # assign initial U0 and U1
    U0 = Matrix(eta,1, complex=A.is_complex) 
    U0.NumPy()[:,0] = 1
    U1 = Matrix(eta,1, complex=A.is_complex) 
    U1.NumPy()[:,0] = 0

    Q.Append(r)

    for j in range(maxiter-1):

        eps = 1e-12 # everything smaller counts as 0

        r.data = A*(Q*U0[:,j])+B*(Q*U1[:,j])
        s = Vector(eta, complex=A.is_complex)
        # MGS: orthogonalize against Q
        for i in range(eta):
            s[i] = InnerProduct(r, Q[i], conjugate=True) 
            r-=s[i]*Q[i]
        alpha = sqrt(InnerProduct(r,r).real)
        w = U0[:,j] 

        # MGS: orthogonalize against V
        for i in range(j+1): 
            T[i,j] = InnerProduct(s, U0[:,i]) + \
                        InnerProduct(w, U1[:,i])
            s -= T[i,j]*U0[:,i]
            w -= T[i,j]*U1[:,i]

        T[j+1,j] = sqrt(alpha*alpha+InnerProduct(s,s).real+
                    InnerProduct(w,w).real)

        # breakdown
        if T[j+1,j].real < eps: 
            print("breakdown in iteration ", j)
            return Q

        # deflation
        if alpha.real < eps:
            print("deflation in iteration ", j)
            tmp = U0
            U0 = Matrix(eta, j+2, complex=A.is_complex) 
            U0[:,:j+1] = tmp
            U0[:,j+1] = 1/T[j+1,j]*s

            tmp = U1
            U1 = Matrix(eta, j+2, complex=A.is_complex) 
            U1[:,:j+1] = tmp
            U1[:,j+1] = 1/T[j+1,j]*w

        else:
            eta+=1
            Q.Append(1/alpha*r)                        

            tmp = U0
            U0 = Matrix(eta, j+2, complex=A.is_complex) 
            U0[:-1,:j+1] = tmp
            U0[:-1,j+1] = 1/T[j+1,j]*s
            U0[eta-1, :] = 0
            U0[eta-1, j+1] = alpha/T[j+1,j]

            tmp = U1
            U1 = Matrix(eta, j+2, complex=A.is_complex) 
            U1[:-1,:j+1] = tmp
            U1[:-1,j+1] = 1/T[j+1,j]*w
            U1[eta-1, j+1] = 0
            U1[eta-1,:j+1] = 0

    return Q