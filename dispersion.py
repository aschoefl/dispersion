from ngsolve import *
from ngsolve.eigenvalues import LOBPCG
# TODO: add to ngsolve.eigenvalues instead
from soar import SOAR
from toar import TOAR 

import scipy
import numpy as np
from netgen.occ import *
try:
    from ngsolve.webgui import Draw as DrawGeo
except:
    from netgen.gui import Draw as DrawGeo
import time
import random
from matplotlib import pyplot as plt
# import plotly.graph_objects as go
# from mayavi import mlab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def Permeability(freq, h0, ms = 1780*1e-4, gamma = 1.75784*1e11):
    if abs(freq) + abs(h0) < 1e-9: 
        return 1.,0.
    o0  = gamma*h0
    om  = gamma*ms
    f = freq*1e9 *2*np.pi
    kappa = om*f/(o0**2-f**2)
    mu = (1+ om*o0/(o0**2-f**2))
    return mu, kappa

def CreateGeometry(air, rod, max_h = 0.2):
    outer = air-rod
    inner = air*rod
            
    outer.faces.name = "outer"
    outer.faces.col=(1,1,0)
        
    inner.faces.col=(1,0,0)
    inner.faces.name="inner"
    shape = Glue([outer, inner])
    
    shape.edges.Max(X).name = "right"
    shape.edges.Max(-X).name = "left"
    shape.edges.Max(Y).name = "top"
    shape.edges.Max(-Y).name = "bot"

    shape.edges.Max(Y).Identify(shape.edges.Min(Y), "bt")
    shape.edges.Max(X).Identify(shape.edges.Min(X), "lr")

    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=max_h))
    mesh.Curve(5)
    DrawGeo (mesh)
    return mesh

def SortIntoBands(values, k_ind, band_structure = False, dist_th = 0.01, ortho_th = 0.7, method = 'dist', plot = False, min_omega = None, max_omega = None):
# method can be 'dist' or 'ortho'

    nzones = len(values)
    val_srt = [None]*nzones

    for zone in range(nzones):
        
        f = np.array(values[zone]['f'])
        k = np.array(values[zone]['k'])
        ev = np.array(values[zone]['ev'])

        if min_omega == None:
            min_omega =  min(f)
        else:
            min_omega = min(min_omega, min(f))
        
        if max_omega == None:
            max_omega =  max(f)
        else:
            max_omega = max(max_omega, max(f))

        # sort lexicographically along k
        ind = np.lexsort((k[:,1],k[:,0]))
        f = f[ind]
        k = k[ind]
        ev = ev[ind]

        bands = [[0]]
        for i in range(1,len(f)):
            if method == 'dist':
                dist = dist_th+1
            elif method == 'ortho':
                dist = 0
            else:
                print("method not defined")
            nearest_band = None
            for j in range(len(bands)):
                b = bands[j]
                if (method=='dist') and (abs(f[b[-1]]-f[i]) < dist):
                    nearest_band = b
                    dist = abs(f[b[-1]]-f[i])
                if (method == 'ortho') and ( abs(InnerProduct(Vector(ev[b[-1]]), Vector(ev[i]))) > dist  ) : 
                    nearest_band = b
                    dist = abs(InnerProduct(Vector(ev[b[-1]]), Vector(ev[i])))
            if (method == 'dist') and (dist > dist_th): 
                # make new band
                bands.append([i])
            elif (method == 'ortho') and (dist < ortho_th): 
                # make new band
                bands.append([i])
            else:
                nearest_band.append(i)

        ret = []
        means = []
        for b in bands:
            tmp = {'f': None, 'k': None, 'ev': None}
            tmp['f'] = f[b]
            tmp['k'] = k[b,:]
            tmp['ev'] = ev[b,:]

            ret.append(tmp)
            
            means.append(np.mean(tmp['f']))
            
        
        # sort bands by mean frequency values
        _, ind, cnt = np.unique(means, return_counts=True, return_index = True)
        for j in range(len(cnt)):
            if cnt[j] > 1:
                means[ind[j]] += 1e-10
        means, ret = zip(*sorted(zip(means, ret)))
        val_srt[zone] = ret

        # print("Found {} bands in zone {}.".format(len(bands), zone))

    if plot:
        if band_structure:
            fig1, ax1 = plt.subplots()
            ax1.set_ylim(min_omega-0.02,max_omega+0.02)
            label = ['Γ', 'X', 'M', 'Γ'] 
            ax1.set_xlim(0,3*np.pi)
            ax1.grid()
            ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
            ax1.set_xticklabels(label)
            ax1.set_ylabel('ωa/(2πc)')
            nbands = min([len(val_srt[i]) for i in range(len(val_srt))])
            for j in range(nbands):
                # Do weighted mean instead
                tmp = (val_srt[0][j]['f'][0]*(val_srt[0][j]['k'][0][0])+val_srt[2][j]['f'][0]*(val_srt[2][j]['k'][0][0]))/(val_srt[0][j]['k'][0][0]+val_srt[2][j]['k'][0][0])

                x_val= np.concatenate([np.array([0]), np.array([k[k_ind[0]] for k in val_srt[0][j]['k']]), 
                                    np.array([k[k_ind[1]] for k in val_srt[1][j]['k']])+np.pi, 
                                    3*np.pi-np.array([k[k_ind[2]] for k in val_srt[2][j]['k']][::-1]), np.array([3*np.pi])])
                y_val = np.concatenate([np.array([tmp]), val_srt[0][j]['f'], val_srt[1][j]['f'], val_srt[2][j]['f'][::-1], np.array([tmp])])

                # x_val= np.concatenate([np.array([k[k_ind[0]] for k in val_srt[0][j]['k']]), 
                #                     np.array([k[k_ind[1]] for k in val_srt[1][j]['k']])+np.pi, 
                #                     3*np.pi-np.array([k[k_ind[2]] for k in val_srt[2][j]['k']][::-1])])
                # y_val = np.concatenate([val_srt[0][j]['f'], val_srt[1][j]['f'], val_srt[2][j]['f'][::-1]])
                ax1.plot(x_val, y_val, '-', markersize = 1.5)
                # ax1.plot(x_val, y_val, 'ko', markersize = 1.5)
                plt.tight_layout()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=nzones) #, figsize=(5, 3))
            ax[0].set_ylabel('ωa/(2πc)')
            for i in range(nzones):
                ax[i].set(xticklabels=[])  # remove the tick labels
                ax[i].tick_params('x', bottom=False)  # remove the ticks
                ax[i].set_ylim(min_omega-0.02,max_omega+0.02)

                if i > 1:
                    ax[i].set(yticklabels=[])  # remove the tick labels
                    ax[i].tick_params('y', left=False)  # remove the ticks
                    # ax[i].set_xlim(k_min, k_max)
                for d in val_srt[i]:
                    ax[i].plot([k[k_ind[i]] for k in d['k']], d['f'], ms=2)  
            for _ax in ax.flat:
                _ax.label_outer()


    return val_srt

class DispersionQEP:
    """ class DispersionQEP

        designed to solve the problem:
        Find a complex scalar lambda  and u in H^1_p such that 
            Integral(mu_mat * (grad(u)+i*k*u) * Conj(grad(v+i*k*v))) = param_omega**2*Integral(eps*u*Conj(v))
        for all v in H^1_p with [kx, ky] = k = p+lambda*s and 
            mu_mat = [[mu, -i*kappa], [i*kappa, mu]]
        for given frequencies param_omega.

        More information can be found in master thesis 'Numerical Computation of Topological Properties of Photonic Crystals' 
        conducted by Amanda Huber at TU Vienna in 2023. 
    """

    method = "TOAR" # can be "SOAR" or "TOAR"
    nr_eigs = 50 # nr of eigenpairs calculated by SOAR or TOAR algorithm
    th_imag = 1e-6 # for requirement abs(lambda.imag) < th_imag
    th_orthonormal = 0.1 # normed eigenvectors v1, v2 are seen as orthonormal if abs(InnerProduct(v1,v2)) < th_orthonormal
    th_norm_append = 1e-6 # if norm(new snaphot vector) < th_norm_append then don't append (after orthogonalizing against other snap vecs)
    accuracy_lambda = 1e-3 # lambda is in range if is in [min_lam-accuracy_lambda, max_lam+accuracy lambda], same if snapshot lambdas are given
    logging = False
    cheap_residual = True

    def __init__(self, mesh, mu, kappa, eps, param_omega , order = 2):
        """ initialize class DispersionQEP

            Input:
            ------
                mesh : netgen.Mesh object
                mu : diagonal elements of permeability tensor; coefficient function that can depend on param_omega
                kappa : off-diagonal elements of permeability tensor; coefficient function that can depend on param_omega
                eps : permittivity; coefficient function that can depend on param_omega
                order : order of fem approximation
        """


        self.fes = Compress(Periodic(H1(mesh, complex=True, order=order)))
        u,v = self.fes.TnT()

        cf_inner = mesh.MaterialCF({"inner":1}, default=0)
        cf_outer = mesh.MaterialCF({"inner":0}, default=1)

        gamma1 = mu/(mu**2-kappa**2)+0*x
        gamma2 = (-1j)*kappa/(mu**2-kappa**2)+0*x


        blf = [
            # blf_kxky2 = -cf_gamma1*u*v * dx
            -cf_inner*u*Conj(v) * dx, -cf_outer*u*Conj(v) * dx, 
            # blf_kx = ( cf_gamma1*(grad(Conj(v))[0]*u -grad(u)[0]*Conj(v)) - cf_gamma2* (grad(u)[1]*Conj(v) + grad(Conj(v))[1]*u) )* dx
            cf_inner*(Conj(grad(v))[0]*u -grad(u)[0]*Conj(v)) * dx,  cf_outer*(Conj(grad(v))[0]*u -grad(u)[0]*Conj(v)) * dx,  -cf_inner*(grad(u)[1]*Conj(v) + Conj(grad(v))[1]*u) * dx,
            # blf_ky = ( cf_gamma1*(grad(Conj(v))[1]*u - grad(u)[1]*Conj(v)) + cf_gamma2*(grad(u)[0]*Conj(v)  + grad(Conj(v))[0]*u) )* dx
            cf_inner*(Conj(grad(v))[1]*u - grad(u)[1]*Conj(v)) * dx,  cf_outer*(Conj(grad(v))[1]*u - grad(u)[1]*Conj(v)) * dx,  cf_inner*(grad(u)[0]*Conj(v)  + Conj(grad(v))[0]*u)* dx,
            #blf_1 = (cf_gamma1*(grad(u)[0]*grad(Conj(v))[0]+grad(u)[1]*grad(Conj(v))[1]) + cf_gamma2*(grad(u)[1]*grad(Conj(v))[0]-grad(u)[0]*grad(Conj(v))[1])) *dx
            cf_inner*(grad(u)[0]*Conj(grad(v))[0]+grad(u)[1]*Conj(grad(v))[1]) * dx,  cf_outer*(grad(u)[0]*Conj(grad(v))[0]+grad(u)[1]*Conj(grad(v))[1]) * dx, cf_inner*(grad(u)[1]*Conj(grad(v))[0]-grad(u)[0]*Conj(grad(v))[1]) *dx,
            # rhs
            (2*np.pi)**2*cf_inner*u*Conj(v) *dx, (2*np.pi)**2*cf_outer*u*Conj(v) *dx
        ]
        
        self.blf = [None]*(len(blf))
        for i in range(len(blf)):
            self.blf[i] = BilinearForm (self.fes)
            self.blf[i] += sum([igl.coef * igl.symbol for igl in blf[i]])
            self.blf[i].Assemble()


        self.param_px = Parameter(0)
        self.param_py = Parameter(0)
        self.param_sx = Parameter(0)
        self.param_sy = Parameter(0)
        self.param_omega = param_omega


        self.qep_blf = {} # dictionary for bilinear forms
        self.coef = {} # dictionary for coefficients that are constant in the domain Omega
        self.ind = {} # dictionary list of indices which blfs are added up for with matrix

        # order of ind: [blf_kxky2 inner g1, blf_kxky2 outer g1, 
        #                   blf_kx inner g1, blf_kx outer g1, blf_kx inner g2, 
        #                   blf_ky inner g1, blf_ky outer g1, blf_ky inner g2, 
        #                   blf_1 inner g1, blf_1 outer g1, blf_1 inner g2,
        #                   rhs_inner, rhs_outer] = [0,1,2,3,4,5,6,7,8,9,10,11,12] 
        self.ind['m'] = [0,1,2,3,4,5,6,7,8,9,10,11,12] 
        self.coef['m'] = {0: -gamma1*(self.param_px**2+self.param_py**2), 1: -(self.param_px**2+self.param_py**2), #blf_kxky2
                          2: gamma1*1j*self.param_px, 3: 1j*self.param_px, 4: gamma2*1j*self.param_px, # blf_kx
                          5: gamma1*1j*self.param_py, 6: 1j*self.param_py, 7: gamma2*1j*self.param_py, # blf_ky
                          8: gamma1, 9: 1+0*x, 10: gamma2, # blf_1
                          11:-self.param_omega**2*eps, 12:-self.param_omega**2 # rhs
                          }
        self.ind['d'] = [0,1,2,3,4,5,6,7]
        self.coef['d'] = {0: 2j*(self.param_sx*self.param_px+self.param_sy*self.param_py) * gamma1, 1: 2j*(self.param_sx*self.param_px+self.param_sy*self.param_py), #blf_kxky2
                          2: self.param_sx * gamma1, 3: self.param_sx, 4: self.param_sx * gamma2, # blf_kx
                          5: self.param_sy * gamma1, 6: self.param_sy, 7: self.param_sy * gamma2 # blf_ky
                          }
        self.ind['k'] = [0,1]
        self.coef['k'] = {0:(self.param_sx**2+self.param_sy**2)*gamma1, 1:(self.param_sx**2+self.param_sy**2)}

        for s in ['m', 'd', 'k']:
            self.qep_blf[s] = BilinearForm (self.fes)
            for i in self.ind[s]:
                self.qep_blf[s] += sum([self.coef[s][i] *igl.coef * igl.symbol for igl in blf[i]])
            self.qep_blf[s].Assemble()

        self.inv = self.qep_blf['m'].mat.Inverse(freedofs=self.fes.FreeDofs(), inverse="umfpack") 

        self.Qred = None

        # just for Chern number calculation
        self.cf_eps = mesh.MaterialCF({"inner":eps}, default=1)

    def _SolveProjectedSmall (self, M, D, K):
        """
        Solves the eigenvalue problem: Find (l, u) such that
            (l**2 * M + l * D + K) * u = 0. 

        Input: 
        ------
            M, D, K : Matrix
        
        Output:
        -------
            lam : Vector containing -1j/l for self.nr_eigs amount of values
            vec : Matrix with u with norm 1 in columns for self.nr_eigs amount of values
        
        """

        half = M.h
        Mi = M.I
        H = Matrix(2*half, 2*half, complex = self.fes.is_complex)

        H[:half, :half] = -Mi*D
        H[:half, half:] = -Mi*K
        H[half:, :half] = Matrix(half, complex=self.fes.is_complex).Identity()
        H[half:, half:] = 0

        lam, eig = scipy.linalg.eig(H, left=False, right = True)

        # from ngsolve import sqrt
        vec = Matrix(eig[0:half,:])
        for i in range(len(vec[0,:])):
            vec[:,i] = 1./sqrt(InnerProduct(vec[:,i], vec[:,i])).real * vec[:,i]
            # for j in range(len(vec[:,0])):
            #     vec[j,i] *= tmp
            # print("IP: ", InnerProduct(vec[:,i], vec[:,i]))
        lam = Vector(-1j/lam)

        return lam, vec 

    def _SolveProjected (self, Q):
        """
        Solves the QEP projected into space spanned by vectors contained in Q.

        Input: 
        ------
            Q : Multivector
        
        Output:
        -------
            output of self._SolveProjectedSmall called with 'small' matrices
        
        """

        M = InnerProduct((self.qep_blf['m'].mat*Q).Evaluate(), Q, conjugate = True)
        D = InnerProduct((self.qep_blf['d'].mat*Q).Evaluate(), Q, conjugate = True)
        K = InnerProduct((self.qep_blf['k'].mat*Q).Evaluate(), Q, conjugate = True)

        return self._SolveProjectedSmall(M, D, K)
    
    def _computeResMat(self):
        """
        Compute matrices for cheap residual calculation.
        """
        
        zeta = {}
        self.__res_mat = {}

        # combinations for <m,m> scalar product wich contains all matrix combinations
        combi = []
        for i in self.ind['m']:
            for j in self.ind['m']:
                if i <= j:
                    combi += [(i,j)]

        with TaskManager():
            for i in self.ind['m']:
                zeta[i] = (self.blf[i].mat*self.Qred).Evaluate()

            for c in combi:
                self.__res_mat[c] = InnerProduct(zeta[c[0]], zeta[c[1]])

        return
    
    def CalculateResidual(self, vec, l):
        """
        Calculated the residual
            Norm( (M + 1j*l*D - l**2*K) * vec ). 

        Input: 
        ------
            vec : small vector
            l : eigenvalue
        
        Output:
        -------
            residual
        
        """
    
        if not self.cheap_residual:
            if not hasattr(self, 'vec_tmp'):
                self.vec_tmp = self.qep_blf['m'].mat.CreateVector()
            self.vec_tmp.data = self.Qred * vec
            return   Norm((self.qep_blf['m'].mat*self.vec_tmp+1j*l*self.qep_blf['d'].mat*self.vec_tmp-l**2*self.qep_blf['k'].mat*self.vec_tmp).Evaluate())
        
        if not hasattr(self, '_DispersionQEP__res_mat'): 
            self._computeResMat()

        def Calc(s1, s2):
            mat = Matrix(len(self.Qred), len(self.Qred), True)
            mat[:] = 0
            # point on mesh to evaluate constant coefficient functions
            mip = self.fes.mesh(self.fes.mesh.vertices[0].point[0], self.fes.mesh.vertices[0].point[1])
            
            for i in self.ind[s1]:
                for j in self.ind[s2]:
                    scalar = self.coef[s1][i](mip)*Conj(self.coef[s2][j])(mip)
                    if i <= j:
                        mat += scalar*self.__res_mat[(i,j)]
                    else:
                        mat += scalar*self.__res_mat[(j,i)].H
            return mat
        
        A = Calc('m', 'm') + l**2*Calc('d', 'd') + l**4*Calc('k', 'k') 
        tmp = Calc('m', 'd')
        A += 1j*l*(tmp.H - tmp)
        tmp = Calc('d', 'k')
        A += 1j*l**3*(tmp.H - tmp)
        tmp = Calc('m', 'k')
        A -= l**2*(tmp.H + tmp)

        # print(sqrt(abs(InnerProduct(vec, A*vec).real))/Norm(A), sqrt(abs(InnerProduct(vec, A*vec).real)))
        
        return sqrt(abs(InnerProduct(vec, A*vec).real)) #abs() because of numerical errors for very small residuals

    def BuildRB (self, params, ps , lrange = [], lparams = [], append = False):
        """
        Build or add to reduced basis (RB) space. 

        Input: 
        ------
            params : parameters omega for which the big system is solved and every eligible solution is added 
            ps : [px, py, sx, sy] such that (kx,ky) = (px, py) + l*(sx, sy) where sx, sy must be 0 or 1
            lrange : list of tuples (lmin, lmax) to determine eligible range for eigenvalues lambda
            lparams : list of values lambda, where only eigenvalues in small environment around these lambda are eligible
            append : expand existing RB space if True and build new RB space if False
        
        """

        if len(lrange) + len(lparams) == 0:
            if self.logging:
                print("in BuildRB: either lrange or lparams must be given")
            return

        if len(lparams) > 0:
            lrange = [(min(lparams),max(lparams))]
            lparams = np.array(lparams)
            # list defined like that to avoid storing references to just one list like it would happen with [[]]*len(..) or [[].copy()]*len(..)
            vecs_snap_l = [[] for _ in range(len(lparams))]
            lams_snap_l = [[] for _ in range(len(lparams))]
            omega_snap_l = [[] for _ in range(len(lparams))]

        if not append or (self.Qred == None):
            self.Qred = None
            self.snapshot_k = []
            self.snapshot_f = []

        self.param_px.Set(ps[0])
        self.param_py.Set(ps[1])
        self.param_sx.Set(ps[2])
        self.param_sy.Set(ps[3])


        def in_range (l):
            ret = False
            for lr in lrange:
                if l.real > lr[0]-self.accuracy_lambda and l.real < lr[1]+self.accuracy_lambda and abs(l.imag) < self.th_imag:
                    ret = True
                    break
            return ret


        def addSnapshot(vec, l, o):

            ''' note: this works better than using half as many complex vectors'''
            hvr = vec.CreateVector()
            hvi = vec.CreateVector()
            for i in range(len(vec)):
                hvr[i] = vec[i].real
                hvi[i] = vec[i].imag
            
            snapshot_added = False

            if Norm(hvr) > 1e-6:
                r_tmp = self.Qred.AppendOrthogonalize (hvr)
                r_tmp = r_tmp[-1]
                # print("real R: \n", r_tmp)
                if abs(r_tmp) < self.th_norm_append:
                    self.Qred = self.Qred[:-1] # TODO: use replace instead
                else:
                    snapshot_added = True
            if Norm(hvi) > 1e-6:
                r_tmp = self.Qred.AppendOrthogonalize (hvi)
                r_tmp = r_tmp[-1]
                if abs(r_tmp) < self.th_norm_append:
                    self.Qred = self.Qred[:-1] # TODO: use replace instead
                else:
                    snapshot_added = True

            if snapshot_added:
                self.snapshot_k += [((self.param_px.Get()+l*self.param_sx.Get()), (self.param_py.Get()+l*self.param_sy.Get()))]
                self.snapshot_f += [o]

                if self.logging:
                    print("add snapshot frequency = {:.5f},  k = {} ".format(self.snapshot_f[-1], self.snapshot_k[-1]))

        with TaskManager():

            for o in params:
                self.param_omega.Set(o)
                if self.logging:
                    print ("frequency =", o, end='\r')
                
                self.qep_blf['m'].Assemble()
                self.qep_blf['d'].Assemble()
                self.qep_blf['k'].Assemble()          

                self.inv.Update()

                A = -self.inv@self.qep_blf['d'].mat
                B = -self.inv@self.qep_blf['k'].mat

                if self.method == "SOAR":
                    Q = SOAR(A,B, self.nr_eigs)
                elif self.method == "TOAR":
                    Q = TOAR(A,B, self.nr_eigs)
                else:
                    print("Error: unknown method")

                lam, vecs = self._SolveProjected(Q)

                Z = (Q * vecs).Evaluate()
                
                if not self.Qred: 
                    self.Qred = MultiVector(Q[0], 0)

                for l, vec in zip(lam, Z):
                    if in_range(l):
                        if len(lparams) > 0: 
                            # sort possible snapshot to respective ks
                            j = np.argmin(abs(lparams-l.real))
                            if abs(lparams[j]-l.real) < self.accuracy_lambda:
                                lams_snap_l[j].append(l)
                                vecs_snap_l[j].append(vec)
                                omega_snap_l[j].append(o)
                        else:
                            addSnapshot(vec, l, o)

            if len(lparams) > 0 :
                for i in range(len(lams_snap_l)):
                    l = lams_snap_l[i]
                    if l == []: 
                        continue
                    vec = vecs_snap_l[i]
                    o = omega_snap_l[i]
                    k = lparams[i]
                    # sort with respect to proximity to k
                    ziped = list(zip(l, vec, o))
                    ziped = sorted(ziped, key= lambda val: abs(val[0]-k) )
                    l, vec, o = zip(*ziped)

                    indices = [0]
                    for j in range(1,len(l)):
                        cnt = 0
                        while cnt < len(indices):
                            add_value = True
                            if abs(InnerProduct(vec[indices[cnt]], vec[j])) > self.th_orthonormal:
                                cnt = len(indices)
                                add_value = False
                            cnt += 1
                        if add_value:
                            indices.append(j)
    
                    for j in indices:
                        addSnapshot(vec[j], l[j], o[j])
            
            # project matrices that do not depend on omega
            self.mat_blf = [None]*len(self.blf) # [kxky2, kx, ky, 1, rhs_inner, rhs_outer]

            for i in range(len(self.blf)):
                self.mat_blf[i] = InnerProduct((self.blf[i].mat*self.Qred).Evaluate(), self.Qred, conjugate = True)

        if self.cheap_residual:
            self._computeResMat()

        if self.logging:
            print("\ndim reduced space = ", len(self.Qred))

    def CalculateValues(self, params, ps, lrange = [], lparams = [], calculate_residual = False):
        """
        Solves the reduced problem for parameters omega contained in 'params'.

        Input: 
        ------
            params : list of parameters omega
            ps : [px, py, sx, sy] such that (kx,ky) = (px, py) + l*(sx, sy) where sx, sy must be 0 or 1
            lrange : list of tuples (lmin, lmax) to determine eligible range for eigenvalues lambda
            lparams : list of values lambda, where only eigenvalues in small environment around these lambda are eligible
            calculate_residual: True or False
        
        Output:
        -------
            dictionary {'f': f, 'k': k, 'ev': ev, 'res':res} with lists 
            'f' : normalized frequencies
            'k' : (kx,ky) = (px, py) + l*(sx, sy)
            'ev' : small eigenvectors
            'res' : residuals   
        """
        
        if not self.Qred:
            print("Error: call BuildRB before CalculateValues")
            return
    
        if len(lrange) + len(lparams) == 0:
            print("in CalculateValues: either lrange or lparams must be given")
            return

        self.param_px.Set(ps[0])
        self.param_py.Set(ps[1])
        self.param_sx.Set(ps[2])
        self.param_sy.Set(ps[3])

        f = []
        k =[]
        ev = []
        res = [] # residual
        
        if len(lparams) > 0:
            lrange = [(min(lparams),max(lparams))]
            lparams = np.array(lparams)
            # list defined like that to avoid storing references to just one list like it would happen with [[]]*len(..) or [[].copy()]*len(..)
            vecs_snap_l = [[] for _ in range(len(lparams))]
            lams_snap_l = [[] for _ in range(len(lparams))]
            omega_snap_l = [[] for _ in range(len(lparams))]
            
        def in_range (l):
            ret = False
            for lr in lrange:
                if l.real > lr[0]-self.accuracy_lambda and l.real < lr[1]+self.accuracy_lambda and abs(l.imag) < self.th_imag:
                    ret = True
                    break
            return ret
        
        def addValue(vec, l, o):
            f.append(o)
            k.append((self.param_px.Get()+l.real*self.param_sx.Get(), self.param_py.Get()+l.real*self.param_sy.Get()))
            ev.append(np.array(vec))
            if calculate_residual:
                res.append(self.CalculateResidual(vec, lamsred[i]))

        with TaskManager():

            for o in params:

                if self.logging:
                    print("frequency = {:.3f}".format(o), end ='\r')
                
                self.param_omega.Set(o)

                # just for expensive residual
                if calculate_residual and not self.cheap_residual:
                    self.qep_blf['m'].Assemble()
                    self.qep_blf['d'].Assemble()
                    self.qep_blf['k'].Assemble()

                # mapped integration point to evaluate constant coefficient functions
                mip = self.fes.mesh(self.fes.mesh.vertices[0].point[0], self.fes.mesh.vertices[0].point[1])

                M = Matrix(len(self.Qred), len(self.Qred), True)
                M[:] = 0
                for i in self.ind['m']:
                    M += self.coef['m'][i](mip) * self.mat_blf[i]

                D = Matrix(len(self.Qred), len(self.Qred), True)
                D[:] = 0
                for i in self.ind['d']:
                    D += self.coef['d'][i](mip) * self.mat_blf[i]

                K = Matrix(len(self.Qred), len(self.Qred), True)
                K[:] = 0
                for i in self.ind['k']:
                    K += self.coef['k'][i](mip) * self.mat_blf[i]

                lamsred, vecsred = self._SolveProjectedSmall(M, D, K)
                
                for i in range(len(lamsred)):
                    l = lamsred[i]
                    if in_range(l):
                        if len(lparams) > 0: 
                            # sort possible values to respective ks
                            j = np.argmin(abs(lparams-l))
                            if abs(lparams[j]-l) < self.accuracy_lambda:
                                lams_snap_l[j].append(l)
                                vecs_snap_l[j].append(vecsred[:,i])
                                omega_snap_l[j].append(o)
                        else:
                            addValue(vecsred[:,i], l, o)
                        
            if len(lparams) > 0 :
                for i in range(len(lams_snap_l)):
                    lams = lams_snap_l[i]
                    if lams == []: 
                        if self.logging:
                            print("no lambdas for l = ", lparams[i])
                        continue
                    vecs = vecs_snap_l[i]
                    os = omega_snap_l[i]
                    # sort with respect to proximity to lparams[i]
                    ziped = list(zip(lams, vecs, os))
                    ziped = sorted(ziped, key= lambda val: abs(val[0]-lparams[i]) )
                    lams, vecs, os = zip(*ziped)

                    indices = [0]
                    for j in range(1,len(lams)):
                        cnt = 0
                        while cnt < len(indices):
                            add_value = True
                            if abs(InnerProduct(vecs[indices[cnt]], vecs[j])) > self.th_orthonormal:
                                cnt = len(indices)
                                add_value = False
                            cnt += 1
                        if add_value:
                            indices.append(j)
    
                    for j in indices:
                        addValue(vecs[j], lams[j], os[j])

        return {'f': f, 'k': k, 'ev': ev, 'res':res}

    def CalculateValuesFull(self, params, ps, lrange = [], calculate_residual = False):
        """
        Solves the big problem for normalized frequencies contained in 'params'.

        Input: 
        ------
            params : list of normalized frequencies
            ps : [px, py, sx, sy] such that (kx,ky) = (px, py) + l*(sx, sy) where sx, sy must be 0 or 1
            lrange : list of tuples (lmin, lmax) to determine eligible range for eigenvalues lambda
            calculate_residual: True or False
        
        Output:
        -------
            dictionary {'f': f, 'k': k, 'ev': ev, 'res':res} with lists 
            'f' : normalized frequencies
            'k' : (kx,ky) = (px, py) + l*(sx, sy)
            'ev' : eigenvectors
            'res' : residuals   
        """
        

        f = []
        k =[]
        ev = []
        res = []

        self.param_px.Set(ps[0])
        self.param_py.Set(ps[1])
        self.param_sx.Set(ps[2])
        self.param_sy.Set(ps[3])

        def in_range (l):
            ret = False
            for lr in lrange:
                if l.real > lr[0]-self.accuracy_lambda and l.real < lr[1]+self.accuracy_lambda and abs(l.imag) < self.th_imag:
                    ret = True
                    break
            return ret

        def addValue(vec, l, o):

            f.append(o)
            k.append((self.param_px.Get()+l*self.param_sx.Get(), self.param_py.Get()+l*self.param_sy.Get()))
            if calculate_residual:
                res.append(Norm((self.qep_blf['m'].mat*vec+1j*l*self.qep_blf['d'].mat*vec-l**2*self.qep_blf['k'].mat*vec).Evaluate()))
            ev.append(vec)
        
        for o in params:
            self.param_omega.Set(o)
            if self.logging:
                print ("frequency =", o, end='\r')
            
            self.qep_blf['m'].Assemble()
            self.qep_blf['d'].Assemble()
            self.qep_blf['k'].Assemble()          

            self.inv.Update()

            A = -self.inv@self.qep_blf['d'].mat
            B = -self.inv@self.qep_blf['k'].mat

            if self.method == "SOAR":
                Q = SOAR(A,B, self.nr_eigs)
            elif self.method == "TOAR":
                Q = TOAR(A,B, self.nr_eigs)
            else:
                print("Error: unknown method")

            lam, vecs = self._SolveProjected(Q)

            Z = (Q * vecs).Evaluate()
            
            for l, vec in zip(lam, Z):
                if in_range(l):
                    addValue(vec, l, o)

        return {'f': f, 'k': k, 'ev': ev, 'res':res}

    def GreedyRB(self, ps, lrange, params, max_it = 50, th_res = 1e-2, nval=None):
        """
        Build a reduced basis (RB) space employing a greedy algorithm. The optimization can be done over multiple zones at once. 
        Each zone is distinguished by a different parameter ps.

        Input: 
        ------
            ps : Define param = [px, py, sx, sy] such that (kx,ky) = (px, py) + l*(sx, sy) where sx, sy must be 0 or 1. 
                Then ps can be of 'param' or list of 'param', one for each zone. 
            lrange : list of tuples (lmin, lmax) to determine eligible range for eigenvalues lambda. 
                Also a list of lists of tuples (lmin, lmax) with same length as 'ps' can be given if the ranges differ for different zones. 
            params : List of frequencies omega from which to choose the snapshots. 
                Also a list of lists of frequencies with same length as 'ps' can be given if the frequencies differ for different zones. 
            max_it : Maximal amount of iterations after which the greedy algorithm stops. 
            th_res : Maximal residual must be smaller than 'th_res' for the algorithm to stop prior to 'max_it' amount of iterations.
            nval : Amount of frequencies that are randomly chosen from 'params' in each step to use in greedy optimization. 
        """

        ACC_LAM_SMALL = 1e-3
        ACC_LAM_BIG = 0.5

        try: 
            ps[0][0]
            nps = len(ps)
        except: 
            nps = 0
        
        # try:
        #     nk_comp = len(k_comp)
        # except:
        #     nk_comp = 0
        
        try:
            lrange[0][0][0]
            nlrange = len(lrange)
        except:
            nlrange = 0

        try:
            params[0][0]
            nparams = len(params)
        except:
            nparams = 0


        nzones = max([1, nps, nlrange, nparams])

        tmp = 0
        zones = [None]*nzones
        for i in range(nzones):
            zones[i] = {}
            if nps == 0:
                zones[i]['ps'] = ps
                if ps[2] == 0:
                    zones[i]['k_comp'] = 1
                else:
                    zones[i]['k_comp'] = 0
            elif nps == nzones:
                zones[i]['ps'] = ps[i]
                if ps[i][2] == 0:
                    zones[i]['k_comp'] = 1
                else:
                    zones[i]['k_comp'] = 0
            else:
                print("invalid input for ps in GreedyRB")
                return
            
            if nlrange == 0:
                zones[i]['lrange'] = lrange
            elif nps == nzones:
                zones[i]['lrange'] = lrange[i]
            else:
                print("invalid input for lrange in GreedyRB")
                return
            
            if nparams == 0:
                zones[i]['params'] = params
            elif nparams == nzones:
                zones[i]['params'] = params[i]
            else:
                print("invalid input for params in GreedyRB")
                return
            tmp += len(zones[i]['params'])

        tmp /= nzones

        original_accuracy_lambda = self.accuracy_lambda

        i_zones_start = 0
        if nval == None:
            nval = round(tmp/5)

        n_initial_omega = 2
        while not self.Qred:
            omega_rb = np.sort([zones[i_zones_start]['params'][i] for i in random.sample(range(len(zones[i_zones_start]['params'])), n_initial_omega)])
            self.BuildRB(omega_rb, zones[i_zones_start]['ps'], lrange=zones[i_zones_start]['lrange'])
            n_initial_omega += 1

        if self.logging:
            print("dim of initial reduced space: ", len(self.Qred))
        for i in range(len(zones)):
            zones[i]['finished'] = False
        begin_time = time.time()
        for iter in range(max_it):

            all_finished = True
            for i in range(len(zones)):
                if not zones[i]['finished']: all_finished = False
            if all_finished: break

            for i in range(len(zones)):

                if zones[i]['finished']: continue

                n_val = min(len(zones[i]['params']), nval)
                if n_val == len(zones[i]['params']):
                    omega_val = np.sort(zones[i]['params'])
                else:
                    omega_val = np.sort([zones[i]['params'][j] for j in random.sample(range(len(zones[i]['params'])), n_val)])
                self.accuracy_lambda = ACC_LAM_SMALL
                zones[i]['values'] = self.CalculateValues(omega_val, zones[i]['ps'], lrange=zones[i]['lrange'], calculate_residual=True)

                cnt = 0
                while not zones[i]['values']['res']:
                    omega_val = np.sort([zones[i]['params'][j] for j in random.sample(range(len(zones[i]['params'])), n_val)])
                    zones[i]['values'] = self.CalculateValues(omega_val, zones[i]['ps'], lrange=zones[i]['lrange'], calculate_residual=True)
                    cnt += 1
                    if cnt > 5:
                        print("no RB space was built") 
                        return 

                residual = zones[i]['values']['res']

                if (max(zones[i]['values']['res']) < th_res): 
                    if self.logging:
                        print("zones {} finished after {} seconds".format(i, time.time()-begin_time))
                    zones[i]['finished'] = True
                    break

                # sort by residual (descending)
                zip_to_sort = list(zip(residual, range(len(residual))))
                sorted_zip = sorted(zip_to_sort, key=lambda x: x[0], reverse=True)
                index = [tup[1] for tup in sorted_zip]

                self.accuracy_lambda = ACC_LAM_BIG
                n_snap_before = len(self.snapshot_f)
                for ind in index:
                    # TODO: deal with zones[i]['ps'][zones[i]['k_comp']+2] == 0
                    l = (zones[i]['values']['k'][ind][zones[i]['k_comp']]-zones[i]['ps'][zones[i]['k_comp']])/zones[i]['ps'][zones[i]['k_comp']+2]
                    self.BuildRB([zones[i]['values']['f'][ind]], zones[i]['ps'], lparams=[l], append=True)
                    
                    if len(self.snapshot_f) > n_snap_before:
                        break
                self.accuracy_lambda = ACC_LAM_SMALL
        
        if iter >= max_it-1:
            print("desired accuracy was not reached in DispersionQEP.GreedyRB")
            
        self.accuracy_lambda = original_accuracy_lambda

        if self.logging:
            print("dim of final reduced space: ", len(self.Qred))

    def CalculateBandStructure(self, params=None, nparam=500, min_omega = 0.0001, max_omega = 0.7, buildRB = True, th_res = 1e-2, nval = 100, useRB = True, plot = True, prefix =''):
        """
        Calculate and (if desired) plot band structure.

        Input: 
        ------
            params : List of parameters omega. If None are given, equidistant parameters are created. 
            nparam : Number of parameters if no parameters are given. 
            min_omega : Minimum of parameters omega. Relevant if no parameters a
            buildRB : if True a new RB space is built
            max_it : Maximal amount of iterations after which the greedy algorithm stops. 
            th_res : Maximal residual must be smaller than 'th_res' for the algorithm to stop prior to 'max_it' amount of iterations.
            nval : Amount of frequencies that are randomly chosen from 'params' in each step to use in greedy optimization. 
            useRB : If True a reduced basis model order reduction is conducted.
            plot : If True the band structure is plotted and the fiure is saved. 
            prefix : Prefix for figure that is saved. Name of figure 'prefix'+'bands.png'

        Output:
        -------
            values : list of return values of CalculateValues or CalculateValuesFull
            times: dictionary with online and offline calculation time
            res: dictionary of min, mean and max residual
        """

        times = {'online': None, 'offline': None}
        
        nzones = 3
        delta = 0
        lrange=[(0, np.pi-delta)]

        # order: Gamma-X – X-M – Gamma-M
        ps = [[0,0,1,0], [np.pi-delta,0,0,1], [0,0,1,1]] 
        lrange=[(0, np.pi-delta)]

        if np.any(params == None):
            params = np.linspace(min_omega, max_omega, nparam)

        if buildRB and useRB:
            tmp = self.th_imag
            self.th_imag = 1e-14
            begin_time = time.time()
            self.GreedyRB(ps, lrange, params= params, th_res=th_res, nval = nval)
            times['offline'] = time.time()-begin_time
            print("RB space of dimension {} built in {} seconds".format(len(self.Qred), times['offline']))
            self.th_imag = tmp

        res_mean = []
        res_min = []
        res_max = []
        values = []

        if useRB:
            begin_time = time.time()
            for i in range(nzones):
                values += [self.CalculateValues(params, ps[i], lrange, calculate_residual=True)]

                res_mean += [np.mean(values[i]['res'])]
                res_min += [min(values[i]['res'])]
                res_max += [max(values[i]['res'])]
            
            times['online'] = time.time()-begin_time
            print("online time: ", times['online'] )
            print("min residual = {:2e}, mean residual = {:2e}, max residual = {:2e}".format(min(res_min), np.mean(res_mean), max(res_max)))
        else:
            begin_time = time.time()
            for i in range(nzones):
                values += [self.CalculateValuesFull(params, ps[i], lrange=lrange, calculate_residual=True)]
                
                res_mean += [np.mean(values[i]['res'])]
                res_min += [min(values[i]['res'])]
                res_max += [max(values[i]['res'])]


            times['offline'] = time.time()-begin_time
            print("calculation time: ", times['offline'])
            print("min residual = {:2e}, mean residual = {:.2e}, max residual = {:2e}".format(min(res_min), np.mean(res_mean), max(res_max)))



        if plot:
            # label = ['ΓX', 'XM', 'MΓ']
            label = [r'$\Gamma$', 'X', 'M', r'$\Gamma$'] 
            fig, ax = plt.subplots()
            ax.set_xlim(0,3*np.pi)
            ax.grid()
            ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
            ax.set_xticklabels( label)
            ax.set_ylim(min(params)-0.02,max(params)+0.02)
            ax.set_ylabel('ωa/(2πc)')

            
            # initialize plots
            # fig_r, ax_r = plt.subplots()
            # fig_v, ax_v = plt.subplots(nrows=1, ncols=3)
        #     fig_r.suptitle('Residual')
        #     fig_v.suptitle('Band Structure')
        #     fig_v.show()
        #     fig_v.canvas.draw()
                
            if useRB:
                f_snap = [[], [], []]
                k_snap = [[], [], []]

                for i in range(len(self.snapshot_k)):
                    k = self.snapshot_k[i]
                    if k[0] == np.pi:
                        f_snap[1] += [self.snapshot_f[i]]
                        k_snap[1] += [self.snapshot_k[i][1]]
                    if k[1] == 0:
                        f_snap[0] += [self.snapshot_f[i]]
                        k_snap[0] += [self.snapshot_k[i][0]]
                    if k[0] == k[1]:
                        f_snap[2] += [self.snapshot_f[i]]
                        k_snap[2] += [self.snapshot_k[i][0]]
                x_snap = np.concatenate([np.array(k_snap[0]), np.array(k_snap[1])+np.pi, 3*np.pi-np.array(k_snap[2])])
                y_snap = np.concatenate([f_snap[0], f_snap[1], f_snap[2]])
            

            x_val= np.concatenate([np.array([k[0] for k in values[0]['k']]), 
                                np.array([k[1] for k in values[1]['k']])+np.pi, 
                                3*np.pi-np.array([k[0] for k in values[2]['k']])])
            y_val = np.concatenate([values[0]['f'], values[1]['f'], values[2]['f']])

            ax.plot(x_val,y_val, 'ko',  markersize=1)
            if useRB:
                ax.plot(x_snap, y_snap, 'r*', markersize=6)

            # , label = "snapshots")
            # fig.legend()
            fig.tight_layout() 
            try:
                plt.savefig("output/{}bands.png".format(prefix), transparent = True)
            except:
                pass

        # for i in range(nzones):
        #     # ax_v[i].set_ylim(min(params)-0.02, max(params)+0.02)
        #     # ax_v[i].set_xlabel(label[i])
        #     # ax_v[i].set(xticklabels=[])  # remove the tick labels
        #     # ax_v[i].tick_params('x', bottom=False)  # remove the ticks
        #     # ax_v[i].plot([k[zones[i]['k_comp']] for k in zones[i]['values']['k']], zones[i]['values']['f'], 'ko', markersize=1)
        #     # if len(k_snap[i]) > 0:
        #     #     ax_v[i].plot(k_snap[i], f_snap[i], 'r*', markersize=4)#, label = "snapshots")
        #     # # ax_v[i].legend()
        #     ax_r.plot(zones[i]['values']['f'], zones[i]['values']['res'], label=label[i])
        
        # ax_r.set_yscale('log')

        # for ax in ax_v.flat:
        #     ax.label_outer()

        # ax_v[0].set_xlim(0, np.pi)
        # ax_v[1].set_xlim(0, np.pi)
        # ax_v[2].set_xlim(np.pi, 0)
        # ax_v[0].set_ylabel("ω/2π")

        res = {'min': res_min, 'mean': res_mean, 'max': res_max}

        return values, times, res

    def ChernNumber_WLA(self, val_srt = None, params = None, nzones = None, buildRB = False, plot = False, prefix = ''):
        
        k_min = -np.pi
        k_max = np.pi

        if val_srt == None:
            if np.any(params == None) or nzones == None:
                print("Either val_srt or params and nzones must be given")
                return
            
            # create zones
            zones = [None]*nzones
            ky = lambda i: k_min+i*(k_max-k_min)/nzones
            for i in range(nzones):
                zones[i] = {'ps': [k_min,ky(i),1,0], 'k_comp': 0, 'lrange': [(0,k_max-k_min)]}    
            
            if buildRB:
                self.GreedyRB(zones=zones, params=params)
                print("RB space of dim {} built".format(len(self.Qred)))
            
            values = [None]*nzones
            for i in range(nzones):
                print("Calculating zone {}/{} with ky = {}".format(i+1, nzones, zones[i]['ps'][1]))
                values[i] = self.CalculateValues(params, ps=zones[i]['ps'], lrange=zones[i]['lrange'])
            val_srt = SortIntoBands(values, k_ind=[zone['k_comp'] for zone in zones], plot = plot)
        else:
            nzones = len(val_srt)
            ky = lambda i: k_min+i*(k_max-k_min)/nzones

        ul = GridFunction(self.fes)
        ur = GridFunction(self.fes)

        if plot:
            tmp1 = np.linspace(k_min, k_max, 500)
            tmp2 = np.linspace(k_min, k_max, 500)
            tmp1, tmp2 =np.meshgrid(tmp1,tmp2)
            a = 6
            b = 15
            X = (b + a*np.cos(tmp1)) * np.cos(tmp2)
            Y = (b + a*np.cos(tmp1)) * np.sin(tmp2)
            Z = a * np.sin(tmp1)

            ky_vec = []
            for i in range(nzones):
                ky_vec += [ky(i)]
            ky_vec += [k_min]
                
        nbands = min([len(val_srt[i]) for i in range(len(val_srt))])
        chern_numbers = []

        for band in range(nbands):

            phase = []
            
            for r in range(nzones):
                
                d = val_srt[r][band]

                result = 1

                for i in range(len(d['f'])):

                    il = i
                    ir = (i+1)%len(d['f'])
                    if i == 0:
                        ul.vec.data = self.Qred*Vector(d['ev'][il, :])
                    else:
                        ul.vec.data = ur.vec.data

                    ur.vec.data  = self.Qred*Vector(d['ev'][ir, :])
                    
                    cfl = CoefficientFunction(exp(1j*(d['k'][il][0]*x+d['k'][il][1]*y))) 
                    cfr = CoefficientFunction(exp(1j*(d['k'][ir][0]*x+d['k'][ir][1]*y)))

                    self.param_omega.Set((d['f'][il]+d['f'][ir])/2)
                    tmp = Integrate (self.cf_eps * ul*cfl*Conj(ur*cfr), self.fes.mesh) #, order=10)
                    result *= tmp/abs(tmp)

                    if abs(tmp) < 1e-8 :
                        print("tmp = 0 in zone ", r )
                        
                phase.append(-np.log(result).imag)

            if plot:
                plt.figure()
                plt.ylim(-np.pi,np.pi)
                for p in range(len(phase)):
                    plt.plot([k_min +p*(k_max-k_min)/nzones], phase[p], 'ko', markersize = 2.5 )
                    plt.xlabel(r'$k_y$')
                    plt.ylabel(r'Berry phase $\phi(k_y)$')
                plt.tight_layout()
                try:
                    plt.savefig("output/{}qep_berry_phase_band{}_nzones{}.png".format(prefix, band, nzones), transparent = True)
                except:
                    pass

                # u1 = np.concatenate([phase,[phase[0]]])
                # v1 = np.array(ky_vec)

                # X1 = (b + (a)*np.cos(u1)) * np.cos(v1)
                # Y1 = (b + (a)*np.cos(u1)) * np.sin(v1)
                # Z1 = a * np.sin(u1)

                # mlab.figure(size = (1024,768), bgcolor = (1,1,1))
                # mlab.mesh(X, Y, Z, opacity=1,  colormap = 'YlOrRd', vmax = b, vmin = -a)
                # mlab.plot3d(X1, Y1, Z1, tube_radius=0.31, color = (0,0,0))# colormap = 'rainbow')
                # # mlab.plot3d(X1, Y1, Z1, tube_radius=0.3, color = (0.7,0.7,0.7) , opacity = 0.5)# colormap = 'rainbow')
                # # mlab.points3d(X1, Y1, Z1, scale_factor = 0.55, color = (0,0,0) )# colormap = 'rainbow')
                # try:
                #     mlab.view(distance=100)
                #     mlab.savefig("{}qep_torus_band{}_nzones{}.png".format(prefix, band, nzones))
                # except:
                #     pass

                # mlab.figure(size = (1024,768), bgcolor = (1,1,1))
                # mlab.mesh(X, Y, Z, opacity=0,  colormap = 'YlOrRd', vmax = b, vmin = -a)
                # mlab.plot3d(X1, Y1, Z1, tube_radius=0.31, color = (0,0,0))# colormap = 'rainbow')
                # # mlab.plot3d(X1, Y1, Z1, tube_radius=0.3, color = (0.7,0.7,0.7) , opacity = 0.5)# colormap = 'rainbow')
                # # mlab.points3d(X1, Y1, Z1, scale_factor = 0.55, color = (0,0,0) )# colormap = 'rainbow')
                # try:
                #     mlab.view(distance=100)
                #     mlab.savefig("{}qep_torus_back_band{}_nzones{}.png".format(prefix, band, nzones))
                # except:
                #     pass

                
            
            chern = 0
            d = lambda p1,p2,m: p2 +2*np.pi*m -p1
            for i in range(len(phase)-1):
                dist = []
                for m in [-1,0,1]:
                    dist += [d(phase[i], phase[i+1], m)]
                j = np.argmin(abs(np.array(dist)))
                chern -= dist[j] 

            chern_numbers +=[round(chern/(2*np.pi))]
            # print("chern number for band {}: {}\n".format(band, round(chern)))

        return chern_numbers

class DispersionGEP:

    th_imag = 1e-6
    cheap_residual = True
    th_norm_append = 1e-6
    logging = False
    
    def __init__(self, mesh, mu, kappa, eps, order = 2):
        """ initialize class DispersionGEP
                designed to solve the problem:
                Find omega and u in H^1_p such that 
                    Integral(mu_mat * (grad(u)+i*k*u) * Conj(grad(v+i*k*v))) = omega**2*Integral(eps*u*Conj(v))
                for all v in H^1_p with 
                    k = [kx, ky], mu_mat = [[mu, -i*kappa], [i*kappa, mu]]
                for certain parameters kx and ky. 

                mesh : netgen.Mesh object
                order : order of fem approximation
        """

        self.fes = Compress(Periodic(H1(mesh, complex=True, order=order)))
        u,v = self.fes.TnT()
        cf_gamma1 = mesh.MaterialCF({"inner":mu/(mu**2-kappa**2)}, default=1)
        cf_gamma2 = mesh.MaterialCF({"inner":(-1j)*kappa/(mu**2-kappa**2)}, default=0)
        self.cf_eps = mesh.MaterialCF({"inner":eps}, default=1)
        
        blf = [
            # blf_kxky2
            -cf_gamma1*u*v * dx,
            # blf_kx
            ((-cf_gamma1*grad(u)[0] - cf_gamma2*grad(u)[1])*v + (cf_gamma1*grad(v)[0] - cf_gamma2*grad(v)[1])*u ) * dx,
            # blf_ky
            ((cf_gamma2*grad(u)[0] - cf_gamma1*grad(u)[1])*v + (cf_gamma2*grad(v)[0] + cf_gamma1*grad(v)[1])*u) * dx,
            # blf_1
            (cf_gamma1*(grad(u)[0]*grad(v)[0]+grad(u)[1]*grad(v)[1]) + cf_gamma2*(grad(u)[1]*grad(v)[0]-grad(u)[0]*grad(v)[1])) *dx,
            # rhs
            (2*np.pi)**2*self.cf_eps*u*v *dx
        ]

        self.blf = [None]*(len(blf))
        for i in range(len(blf)):
            self.blf[i] = BilinearForm (self.fes)
            self.blf[i] += sum([igl.coef * igl.symbol for igl in blf[i]])
            self.blf[i].Assemble()

        self.param_kx = Parameter(1)
        self.param_ky = Parameter(1)
        self.param_eval = Parameter(1) # for residual
        self.Qred = None

        # coefficients for the bilinear forms that are constant in the domain
        self.coef = [-(self.param_kx**2+self.param_ky**2), 1j*self.param_kx , 1j*self.param_ky, 1+0*x, -self.param_eval]
        self.m = BilinearForm (self.fes)
        for i in range(len(blf)-1):
            self.m += sum([self.coef[i] * igl.coef * igl.symbol for igl in blf[i]])
        self.pre = Preconditioner(self.m, "direct")
        self.m.Assemble()
   
    def _computeResMat(self):

            zeta = {}
            self.__res_mat = {}

            # combinations for <m,m> scalar product wich contains all matrix combinations
            combi = []
            for i in range(len(self.blf)):
                for j in range(len(self.blf)):
                    if i <= j:
                        combi += [(i,j)]

            with TaskManager():
                for i in range(len(self.blf)):
                    zeta[i] = (self.blf[i].mat*self.Qred).Evaluate()

                for c in combi:
                    self.__res_mat[c] = InnerProduct(zeta[c[0]], zeta[c[1]])#.T
            return

    def BuildRB (self, params, nbands, append = False, bands=None):
        """
        build reduced basis for snapshot parameters given in snapshot (contains tuples (kx,ky))
        """

        if not append:
            self.Qred = None
            self.snapshot_k = []
            self.snapshot_f = []
        
        with TaskManager():

            for s in params:
                if self.logging:
                    print ("snapshot =", s)

                # set parameters
                self.param_kx.Set(s[0])
                self.param_ky.Set(s[1])
                self.m.Assemble()

                # solve eigenvalue problem for first nbands eigenvalues
                # evals, evecs = PINVIT_new(self.m.mat,self.blf[4].mat,self.pre, num = nbands, maxit=20, printrates=False)
                evals, evecs = LOBPCG(self.m.mat,self.blf[4].mat,self.pre, num = nbands, maxit=20, printrates=False)
                

                for vec in evecs: vec /= Norm(vec)  

                if not self.Qred: self.Qred = MultiVector(evecs[0], 0)
                    
                snapshot_added = False
                freq = [None]*len(evals)
                for j in range(len(evals)):

                    if bands:
                        if not j in bands: continue
                        
                    vec = evecs[j]
                    if evals[j] < -1e-6:
                        print ("negative eigenvalue ", evals[j])
                        continue

                    ''' note: this works better than using half as many complex vectors'''
                    hvr = vec.CreateVector()
                    hvi = vec.CreateVector()
                    for i in range(len(vec)):
                        hvr[i] = vec[i].real
                        hvi[i] = vec[i].imag

                    if Norm(hvr) > 1e-6:
                        r_tmp = self.Qred.AppendOrthogonalize (hvr)
                        r_tmp = r_tmp[-1]
                        if abs(r_tmp) < self.th_norm_append:
                            self.Qred = self.Qred[:-1] # TODO: use replace instead 
                        else:
                            freq[j] = np.sqrt(np.abs(evals[j]))
                            snapshot_added = True
                    if Norm(hvi) > 1e-6:
                        r_tmp = self.Qred.AppendOrthogonalize (hvi)
                        r_tmp = r_tmp[-1]
                        if abs(r_tmp) < self.th_norm_append:
                            self.Qred = self.Qred[:-1] # TODO: use replace instead
                        else:
                            freq[j] = np.sqrt(np.abs(evals[j]))
                            snapshot_added = True
                    # self.Qred.AppendOrthogonalize(vec)
                if snapshot_added:
                    self.snapshot_k += [s]
                    self.snapshot_f += [np.array(freq)]
                    if self.logging:
                        print("snapshot frequencies: ", freq)

                else:
                    if self.logging:
                        print("not added")

            # calculate reduced matrices
            self.red = [None]*len(self.blf)
            for i in range(len(self.red)):
                # self.red[i] = InnerProduct(self.Qred, self.blf[i].mat*self.Qred, conjugate = True)
                self.red[i] = InnerProduct((self.blf[i].mat*self.Qred).Evaluate(), self.Qred, conjugate = True)


            if self.cheap_residual:
                self._computeResMat()

        if self.logging:
            print("dim reduced space = ", len(self.Qred))
        return
   
    def CalculateValues(self, params, nbands, calculate_residual = False):
        """ Calculate nbands many eigenvalues and eigenvectors for k=(kx,ky) in params.

            Return: 
            ------
            - f: frequency (f = omega)
            - ev: eigenvectors
        """

        if not self.Qred:
            raise Exception("Error: call BuildRB before CalculateValues")
        
            
        f = []
        k_ret =[]
        ev = []
        res = [] # residual

        if calculate_residual:
            if self.cheap_residual:
                if not hasattr(self, '_DispersionGEP__res_mat'): 
                    self._computeResMat()
                mat = Matrix(len(self.Qred), len(self.Qred), True)
                mip = self.fes.mesh(self.fes.mesh.vertices[0].point[0], self.fes.mesh.vertices[0].point[1])
            else:
                vec_res = self.m.mat.CreateVector()

        with TaskManager():

            for k in params:

                self.param_kx.Set(k[0])
                self.param_ky.Set(k[1])

                k_ret.append(k)
                if self.logging:
                    print ("(kx,ky) = ({:.2f},{:.2f})".format(k[0], k[1]), end='\r')

                # point on mesh to evaluate constant coefficient functions
                mip = self.fes.mesh(self.fes.mesh.vertices[0].point[0], self.fes.mesh.vertices[0].point[1])

                mat = Matrix(len(self.Qred), len(self.Qred), True)
                for i in range(4):
                    mat += self.coef[i](mip) * self.red[i]

                evals, evecs = scipy.linalg.eig(a = mat,b = self.red[4])

                if calculate_residual:
                    if not self.cheap_residual:
                        self.m.Assemble()
                
                ind = []
                # discard eigenvalues with not nearly zero imaginary part # TODO: not nessesary bc hermitean
                for i in range(len(evals)):
                    if abs (evals[i].imag) < self.th_imag:
                        ind += [i]
                    else:
                        print('here at (kx,ky) = ({:.2f},{:.2f}) with evals[i].imag = {}'.format(k[0], k[1], evals[i].imag))
                
                # choose nbands smalles eigenvalues
                _, ind = zip(*sorted(zip( evals[ind], ind)))
                f.append(np.array([np.sqrt(abs(evals[i].real)) for i in ind[0:nbands]]))
                
                # choose associated eigenvectors
                ''' TODO: save as MV? '''
                tmp = []
                res_tmp = []
                for i in ind[0:nbands]:
                    vec = Vector(len(evecs[:,0]), complex = self.fes.is_complex)
                    vec.NumPy()[:] = evecs[:,i]
                    if calculate_residual:
                        if self.cheap_residual:
                            self.param_eval.Set(evals[i].real)
                            mat[:] = 0
                            
                            for i in range(len(self.blf)):
                                for j in range(len(self.blf)):
                                    scalar = self.coef[i](mip)*Conj(self.coef[j])(mip)
                                    if i <= j:
                                        mat += scalar*self.__res_mat[(i,j)]
                                    else:
                                        mat += scalar*self.__res_mat[(j,i)].H
                            
                            res_tmp.append(sqrt(abs(InnerProduct(vec, mat*vec).real)))
                            # print(np.sqrt(res_tmp[-1]), res_tmp[-1])
                        else:
                            vec_res.data = self.Qred*vec
                            res_tmp.append(Norm((self.m.mat*vec_res-evals[i]*self.blf[4].mat*vec_res).Evaluate()))
                    tmp.append(vec)
                ev.append(tmp)
                if calculate_residual:
                    res.append(res_tmp)

        return {'f': f, 'k': k_ret, 'ev': ev, 'res':res}

    def CalculateValuesFull(self, params, nbands, calculate_residual = False):

            f = []
            k_ret =[]
            ev = []
            res = [] # residual

            with TaskManager():

                for k in params:

                    self.param_kx.Set(k[0])
                    self.param_ky.Set(k[1])
                    k_ret.append(k)
                    if self.logging:
                        print ("(kx,ky) = ({:.2f},{:.2f})".format(k[0], k[1]), end='\r')

                    self.m.Assemble()

                    evals, evecs = LOBPCG(self.m.mat,self.blf[4].mat,self.pre, num = nbands, maxit=20, printrates=False)

                    for vec in evecs: vec /= Norm(vec)  

                    tmp = []
                    for i in range(nbands):
                        if evals[i] > -1e-6:
                            tmp += [np.sqrt(abs(evals[i]))]
                        else:
                            print(evals[i])
                    f.append(np.array(tmp))
                    ev.append(evecs[:nbands])

                    if calculate_residual: 
                        res_tmp = []
                        for j in range(nbands):
                            res_tmp.append(Norm((self.m.mat*evecs[j]-evals[j]*self.blf[4].mat*evecs[j]).Evaluate()))
                        res.append(res_tmp)
                        
            return {'f': f, 'k': k_ret, 'ev': ev, 'res':res}
        
    def GreedyRB(self, params, nbands, max_it = 50, th_res = 1e-2, nval=None):

        if nval == None:
            nval = round(len(params)/5)
        else:
            nval = min(len(params), nval)

        n_initial_k = 2
        while not self.Qred:
            k_rb = [params[i] for i in random.sample(range(len(params)), n_initial_k)]
            self.BuildRB(k_rb, nbands, bands=random.sample(range(nbands),1))
            n_initial_k += 1

        if self.logging: 
            print("dim of initial reduced space: ", len(self.Qred))

        begin_time = time.time()
        for iter in range(max_it):
            
            if nval == len(params):
                k_rb = params
            else:
                k_rb = [params[i] for i in random.sample(range(len(params)), nval)]
            values = self.CalculateValues(k_rb, nbands, calculate_residual=True)
            flat_k = []
            flat_bands = []
            flat_res = []
            
            for i in range(len(values['k'])):
                for j in range(nbands):
                    if j >= len(values['res'][i]): continue
                    flat_k += [values['k'][i]]
                    flat_bands += [j]
                    flat_res += [values['res'][i][j]]
                
            residual = flat_res

            if self.logging:
                print("max res: ", max(residual))
            
            if (max(residual) < th_res):
                if self.logging:
                    print("Finished after {} seconds".format(time.time()-begin_time))
                break

            # sort by residual (descending)
            zip_to_sort = list(zip(residual, range(len(residual))))
            sorted_zip = sorted(zip_to_sort, key=lambda x: x[0], reverse=True)
            index = [tup[1] for tup in sorted_zip]
            
            n_snap_before = len(self.snapshot_k)
            for ind in index:
                self.BuildRB([flat_k[ind]], nbands, append=True, bands = [flat_bands[ind]])
                if len(self.snapshot_k) > n_snap_before:
                    break
        
        if iter >= max_it-1:
            print("desired accuracy was not reached in DispersionGEP.GreedyRB")

        if self.logging:
            print("dim of final reduced space: ", len(self.Qred))

    def CalculateBandStructure(self, nbands, nparam=100, nval = 100, buildRB = True, th_res = 1e-3, min_omega = 0, max_omega = 0.7, useRB = True, plot = True, prefix=''):
        
        times = {'online': None, 'offline': None}
        param = []
        # added floats for display purposes 
        nparam = nparam/(2+sqrt(2))
        for kx in np.linspace(0+1e-10, np.pi, round(nparam)):
            param += [(kx, 0.)]
        for ky in np.linspace(0, np.pi-1e-10, round(nparam)):
            param += [(np.pi, ky)]
        for kxy in np.linspace(0, np.pi, round(nparam*sqrt(2))):
            param += [(kxy, kxy)]

        # nparam = nparam/(2+sqrt(2))
        # for kx in np.linspace(0, np.pi, round(nparam)):
        #     param += [(kx, 0.)]
        # for ky in np.linspace(0, np.pi, round(nparam)):
        #     param += [(np.pi, ky)]
        # for kxy in np.linspace(0, np.pi, round(nparam*sqrt(2))):
        #     param += [(kxy, kxy)]

        if buildRB and useRB:
            begin_time = time.time()
            self.GreedyRB(param, nbands, th_res=th_res, nval = nval)
            times['offline'] = time.time()-begin_time
            print("RB space of dimension {} built in {:.3f} seconds".format(len(self.Qred), times['offline']))
        
        if useRB:
            begin_time = time.time()
            values = self.CalculateValues(param, nbands, calculate_residual=True)
            times['online'] = time.time()-begin_time
            print("online time: {:.3f} seconds".format(times['online'] ))
        else:
            begin_time = time.time()
            values = self.CalculateValuesFull(param, nbands, calculate_residual=True)
            times['offline'] = time.time()-begin_time
            print("calculation time: {:.3f} seconds".format(times['offline']))


        if plot:

            f_val = [[], [], []]
            k_val = [[], [], []]

            for i in range(len(values['k'])):
                k=values['k'][i]
                if k[1] == 0 and k[0] > 0:
                    f_val[0] += [values['f'][i]]
                    k_val[0] += [values['k'][i][0]]
                if k[0] == np.pi:
                    f_val[1] += [values['f'][i]]
                    k_val[1] += [values['k'][i][1]]
                if k[0] == k[1]:
                    f_val[2] += [values['f'][i]]
                    k_val[2] += [values['k'][i][0]]


            if useRB:
                f_snap = [[], [], []]
                k_snap = [[], [], []]

                for i in range(len(self.snapshot_k)):
                    k = self.snapshot_k[i]
                    if k[0] == np.pi:
                        f_snap[1] += [self.snapshot_f[i]]
                        k_snap[1] += [self.snapshot_k[i][1]]
                    if k[1] == 0:
                        f_snap[0] += [self.snapshot_f[i]]
                        k_snap[0] += [self.snapshot_k[i][0]]
                    if k[0] == k[1]:
                        f_snap[2] += [self.snapshot_f[i]]
                        k_snap[2] += [self.snapshot_k[i][0]]
                        
            


            if (min_omega == None ) and (max_omega == None):
                for i in range(0,3):
                    if min_omega == None:
                        min_omega = min([min(tmp) for tmp in f_val[i]])    
                    else:
                        min(min_omega, min([min(tmp) for tmp in f_val[i]]))
                    if max_omega == None:
                        max_omega = max([max(tmp) for tmp in f_val[i]])    
                    else:
                        max(max_omega, max([max(tmp) for tmp in f_val[i]]) )


            # label = ['ΓX', 'XM', 'MΓ']
            # fig, ax = plt.subplots(nrows=1, ncols=3)
            # ax[0].set_xlim(0, np.pi)
            # ax[1].set_xlim(0, np.pi)
            # ax[2].set_xlim(np.pi, 0)
            # for i in range(0,3):
            #     ax[i].set_ylim(min_omega-0.02,max_omega+0.02)
            #     ax[i].plot(k_val[i],f_val[i], 'o',  markersize=1) # 'ko'
            #     ax[i].set_xlabel(label[i])
            #     ax[i].set(xticklabels=[])  # remove the tick labels
            #     ax[i].tick_params('x', bottom=False)  # remove the ticks
            #     ax[i].plot(k_snap[i], f_snap[i], 'r*', markersize=6)#, label = "snapshots")

            # for ax in ax.flat:
            #     ax.label_outer()

            # fig.tight_layout() 
                
            label = [r'$\Gamma$', r'$X$', 'M', r'$\Gamma$'] 
            fig, ax = plt.subplots()
            ax.set_xlim(0,3*np.pi)
            ax.grid()
            ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
            ax.set_xticklabels(label)
            
            # ax.set_ylabel('ωa/(2πc)')
            ax.set_ylabel('ωa/(2πc)')

            # 
            if useRB:
                # x_val= np.concatenate([np.array(k_val[0]), np.array(k_val[1])+np.pi, 3*np.pi-np.array(k_val[2])])
                # y_val = np.concatenate([f_val[0], f_val[1], f_val[2]])
    
                # TODO: this way also for QEP
                ax.plot(np.array(k_val[0]),f_val[0], 'ko', markersize=1)
                plt.gca().set_prop_cycle(None)
                ax.plot(np.array(k_val[1])+np.pi,f_val[1], 'ko', markersize=1)
                plt.gca().set_prop_cycle(None)
                ax.plot(3*np.pi-np.array(k_val[2]),f_val[2], 'ko', markersize=1)

                # ax.plot(x_val,y_val, 'ko',  markersize=1)
                # x_snap = np.concatenate([np.array(k_snap[0]), np.array(k_snap[1])+np.pi, 3*np.pi-np.array(k_snap[2])])
                # y_snap = np.concatenate([f_snap[0], f_snap[1], f_snap[2]])
                
                ax.plot(k_snap[0],f_snap[0], 'r*', markersize=6)
                ax.plot( np.array(k_snap[1])+np.pi,f_snap[1], 'r*', markersize=6)
                ax.plot(3*np.pi-np.array(k_snap[2]), f_snap[2], 'r*', markersize=6)

                # ax.plot(x_snap, y_snap, 'r*', markersize=6)
            else:
                # ax.plot(np.array(k_val[0]),f_val[0],  markersize=1.5)
                # plt.gca().set_prop_cycle(None)
                # ax.plot(np.array(k_val[1])+np.pi,f_val[1],  markersize=1.5)
                # plt.gca().set_prop_cycle(None)
                # ax.plot(3*np.pi-np.array(k_val[2]),f_val[2], markersize=1.5)

                ax.plot(np.array(k_val[0]),f_val[0], 'ko', markersize=1)
                plt.gca().set_prop_cycle(None)
                ax.plot(np.array(k_val[1])+np.pi,f_val[1], 'ko', markersize=1)
                plt.gca().set_prop_cycle(None)
                ax.plot(3*np.pi-np.array(k_val[2]),f_val[2], 'ko', markersize=1)

                # , label = "snapshots")
            # fig.legend()

            ax.set_ylim(min_omega-0.02,max_omega+0.02)
            fig.tight_layout() 

            try:
                plt.savefig("output/{}bands.png".format(prefix), transparent = True)
            except:
                pass

            # plt.figure()
            # plt.plot(values['res'])
            # plt.yscale('log')
            # plt.show()

        res = {'min': np.min(values['res']), 'mean': np.mean(values['res']), 'max': np.max(values['res'])}
        print("min residual: {:.2e}, mean residual: {:.2e}, max residual: {:.2e}".format(np.min(values['res']), np.mean(values['res']), np.max(values['res'])))
        return values, times, res
    
    def ChernNumber_FPC(self, nbands, ngrid, values = None, full = False, buildRB = False, plot = False, prefix = ''):

        k_min = -np.pi
        k_max = np.pi

        kx = lambda i: k_min+(i%ngrid)*(k_max-k_min)/ngrid
        ky = lambda i: k_min+(int(i/ngrid))*(k_max-k_min)/ngrid
        params =[]
        for i in range(ngrid**2):
            params += [(kx(i), ky(i))]

        if buildRB:
            self.GreedyRB(params, nbands)
        
        if values == None and not full:
            values = self.CalculateValues(params, nbands, calculate_residual = False)
            # print("mean residual: ", np.mean(values['res']))
        
        if values == None and full:
            values = self.CalculateValuesFull(params, nbands, calculate_residual = False)

        ul = GridFunction(self.fes)
        ur = GridFunction(self.fes)

        begin_time = time.time()
        chern_numbers = []
        for band in range(nbands):

            kx_vec = []
            ky_vec = []
            phi_vec = []

            chern = 0

            for i in range(ngrid*ngrid):

                kx_vec += [kx(i)]
                ky_vec += [ky(i)]

                cx = 1
                cy = ngrid
                if i%ngrid == ngrid-1:
                    cx = -(ngrid-1)
                if int(i/ngrid)==ngrid-1:
                    cy = -(ngrid-1)*ngrid

                nodes = [i,i+cx,i+cy+cx,i+cy]

                result = 1
                for j in range(len(nodes)):
                    ind_l = nodes[j]
                    ind_r = nodes[(j+1)%len(nodes)]

            #         print("l: (kx, ky) = ({},{}), r: (kx, ky) = ({}, {})".format(kx(ind_l), ky(ind_l), kx(ind_r), ky(ind_r)))

                    if not full:
                        try:
                            ul.vec.data = self.Qred*values['ev'][ind_l][band]
                            ur.vec.data  = self.Qred*values['ev'][ind_r][band]
                        except:
                            return ['-']*nbands
                    else:
                        # try:
                        ul.vec.data = values['ev'][ind_l][band]
                        ur.vec.data  = values['ev'][ind_r][band]
                        # except:
                            # print(values['f'][ind_l][band],values['f'][ind_r][band])
                            # quit()
                        

                    cfl = CoefficientFunction(exp(1j*(kx(ind_l)*x+ky(ind_l)*y))) 
                    cfr = CoefficientFunction(exp(1j*(kx(ind_r)*x+ky(ind_r)*y)))

                    tmp = Integrate (self.cf_eps * ul*cfl*Conj(ur*cfr), self.fes.mesh) 
                    result *= tmp/abs(tmp)
                    
                phi_vec += [-np.log(result).imag]
                chern -= np.log(result).imag

            # print("chern number for band {}: {} ".format(band, round(chern/(2*np.pi))))

            if plot:
                X  = np.reshape(kx_vec, (ngrid, ngrid))
                Y  = np.reshape(ky_vec, (ngrid, ngrid))
                Z  = np.reshape(phi_vec, (ngrid, ngrid))

                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot_surface(X, Y, Z, cmap='rainbow')
                # ax.zaxis.set_major_formatter(FormatStrFormatter('%.1e'))
                plt.locator_params(axis='x', nbins=7)
                plt.locator_params(axis='y', nbins=7)
                ax.tick_params(axis='z', which='major', pad=15)
                
                ax.set_ylabel(r'$k_y$', labelpad=10)
                ax.set_xlabel(r'$k_x$', labelpad=10)
            #     ax.set_zlabel('Berry curvature', labelpad=30)
                
            #     plt.locator_params(axis='z', nbins=5)

            #     ticks = np.linspace(-3, 3, 5)
            #     ax.set_yticks(ticks)
                # if first:
                #     first = False
                #     minz = -0.0001
                #     maxz = 0.0001
                #     ax.set_zlim(minz, maxz)
                
            #     fig1, ax = plt.subplots()
            #     ax.set_box_aspect(1)

            #     cf = ax.contourf(Z, cmap=cm.rainbow)
            # #     cf = ax.matshow(Z, cmap=cm.rainbow)
            #     fig1.colorbar(cf, format='%.1e')
                plt.tight_layout()
                try:
                    # plt.savefig("../output/berry_curvature_band{}_ngrid{}.png".format(band, ngrid))
                    plt.savefig("output/"+prefix+"berry_curvature_band{}_ngrid{}.png".format(band, ngrid))
                except:
                    pass
        #         minz = -2.5*np.pi/ngrid
        #         maxz = 1*np.pi/ngrid
                
        #         fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, coloraxis='coloraxis',
        #             contours = {"z": {"show": True, "start":minz, "end": maxz, "size": 0.005}},                     
        #             )])
        #         camera = dict(
        #             up=dict(x=0, y=0, z=1),
        #             center=dict(x=0, y=0, z=0),
        # #             eye=dict(x=1.25, y=1.25, z=1.25)
        #             eye=dict(x=1.4, y=1.4, z=1.4)
                    
        #         )
        #         fig.update_layout(
        #             font = dict(size = 18),
        #             scene = dict(
        #                 zaxis = dict(range=[minz,maxz], 
        #                     nticks=8,
        #                     title=dict(text="Berry Curvature", font=dict(size=26)),
        #                 ),
        #                 xaxis=dict( 
        # #                     nticks = ngrid+1,
        #                     title=dict(text="kx", font=dict(size=26)),
        #                 ),
        #                 yaxis=dict(
        # #                     nticks = ngrid+1,
        #                     title=dict(text="ky", font=dict(size=26)),
        #                 ),
        #                 aspectmode = 'cube',
        # #                 zaxis_title='Berry Curvature',
                        
        #             ),
        #             coloraxis_colorbar_len=0.5,
        #             coloraxis_colorbar_thickness=30,
        #             coloraxis_colorbar_tickfont = dict(size = 26),
        #             colorscale = dict( sequential = 'Rainbow', diverging = "Rainbow", sequentialminus = 'Rainbow'),
        #             coloraxis = dict(cmin = minz, cmax = maxz), 
        #             width=1000,
        #             height = 1000,
        #             margin=dict(r=1, b=1, l=1, t=1),
        #             scene_camera = camera,
        #         )
                
        #         # fig.show(render = 'png')
        #         fig.write_image("../output/berry_curvature_band{}_ngrid{}.png".format(band, ngrid))


                # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                # surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,  linewidth=0, antialiased=False, vmin=-np.pi/ngrid, vmax=np.pi/ngrid)
                # ax.set_zlim(-2*np.pi/ngrid, 2*np.pi/ngrid)
                # fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.13)
                # ax.set_xlabel(r'$k_x$', labelpad = 7)
                # ax.set_ylabel(r'$k_y$', labelpad = 9)
                # ax.set_zlabel(r'$\Omega(\mathbf{k})$' , labelpad = 7)
                # fig.tight_layout()

                # plt.show()
            
            chern_numbers += [round(chern/(2*np.pi))]
        
        print("FPC: Chern numbers {}, calculated in {:.3f} seconds".format(chern_numbers, time.time()-begin_time))
        return chern_numbers
    
    def ChernNumber_WLA(self, nbands, nzones, nval_per_zone, values = None, full = False, buildRB = False, plot = False, prefix='', extra = False):

        k_min = -np.pi
        k_max = np.pi

        kx = lambda i: k_min+(i%nval_per_zone)*(k_max-k_min)/nval_per_zone
        ky = lambda i: k_min+(int(i/nval_per_zone))*(k_max-k_min)/nzones
        params =[]
        for i in range(nval_per_zone*nzones):
            params += [(kx(i), ky(i))]

        
        if buildRB:
            self.GreedyRB(params, nbands)

        if values == None and not full:
            values = self.CalculateValues(params, nbands, calculate_residual = False)

        if values == None and full:
            values = self.CalculateValuesFull(params, nbands, calculate_residual = False)

        ul = GridFunction(self.fes)
        ur = GridFunction(self.fes)

        chern_numbers = []
        if extra:
            phases = []

        begin_time = time.time()

        for band in range(nbands):

            phase = []

            for r in range(nzones):

                result = 1

                for i in range(nval_per_zone):

                    il = i+r*nval_per_zone
                    ir = (i+1)%nval_per_zone+r*nval_per_zone

                    if not full: 
                        ul.vec.data = self.Qred*values['ev'][il][band]
                        ur.vec.data  = self.Qred*values['ev'][ir][band]
                    else:
                        ul.vec.data = values['ev'][il][band]
                        ur.vec.data  = values['ev'][ir][band]

                    cfl = CoefficientFunction(exp(1j*(ky(r*nval_per_zone)*y+kx(il)*x))) 
                    cfr = CoefficientFunction(exp(1j*(ky(r*nval_per_zone)*y+kx(ir)*x)))
                    
                    tmp = Integrate (self.cf_eps * ul*cfl*Conj(ur*cfr), self.fes.mesh) 
                    if abs(tmp) < 1e-10:
                        print("tmp = 0 in zone ", r )
                        return ['-']*nbands
                    else:
                        result *= tmp/abs(tmp)

                phase.append(-np.log(result).imag)

            if plot:
                plt.figure()
                plt.ylim(-np.pi,np.pi)
                for p in range(len(phase)):
                    plt.plot([k_min +p*(k_max-k_min)/nzones], phase[p], 'ko', markersize = 2.5 )
                    plt.xlabel(r'$k_y$')
                    plt.ylabel(r'Berry phase $\phi(k_y)$')
                plt.tight_layout()
                try:
                    # plt.savefig("../output/berry_phase_band{}_nzones{}.png".format(band, nzones), transparent = True)
                    plt.savefig('output/'+prefix+"berry_phase_band{}_nzones{}.png".format(band, nzones), transparent = True)
                except:
                    pass

            chern = 0
            d = lambda p1,p2,m: p2 +2*np.pi*m -p1
            for i in range(len(phase)-1):
                dist = []
                for m in [-1,0,1]:
                    dist += [d(phase[i], phase[(i+1)%len(phase)], m)]
                j = np.argmin(abs(np.array(dist)))
                chern -= dist[j]

            chern_numbers += [round(chern/(2*np.pi))]
            if extra:
                phases += [np.array(phase)]
            # print("chern number for band {}: {}\n".format(band, round(chern)))

        print("WLA: Chern numbers {}, calculated in {:.3f} seconds".format(chern_numbers, time.time()-begin_time))

        if not extra:
            return chern_numbers
        else:
            return chern_numbers, phases
