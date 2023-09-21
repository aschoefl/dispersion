from ngsolve import *
import numpy as np

def SOAR (A, B, maxiter=50, start=None):

    eps = 1e-12 # everything smaller counts as 0

    r = A.CreateVector()
    f = A.CreateVector()
    f[:] = 0

    q = A.CreateVector()
    if start:
        q.data = start
    else:
        q.SetRandom()
        q /= sqrt(InnerProduct(q,q).real)

    T = Matrix(maxiter, complex=A.is_complex)

    T[:,:] = 0

    Q = MultiVector(q,0)
    F = MultiVector(f,0)
    
    for j in range(maxiter-1):
        Q.Append(q)
        r.data = A*q + B*f

        for i in range(j+1):
            T[i,j] = InnerProduct(r, Q[i], conjugate = True)
            r -= T[i,j]*Q[i]
        
        T[j+1,j] = sqrt(InnerProduct(r,r).real)
        if T[j+1,j].real < eps:
            print("T[j+1,j] == 0 in iteration ", j)
            T[j+1,j] = 1
            q[:] = 0
            f.data = Q*T[1:j+2,:j+1].I[:,j]
            F.AppendOrthogonalize(f)
            if (Norm(F[len(F)-1]) < eps):
                print("breakdown in iteration ", j)
                return Q
            else:
                print("deflation in iteration ", j)
        else:
            q.data = 1./T[j+1,j]*r
            # QUESTION: is there a way to use the fact that T[1:,:j+1] is upper triangle?
            f.data = Q*T[1:j+2,:j+1].I[:,j]

    Q.Append(q)
    return Q
