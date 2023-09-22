from ngsolve import *
import numpy as np
from dispersion import *
import matplotlib.pyplot as plt
from netgen.occ import *

th_res = 1e-3
order = 3
nval = 50
nparam = 500

nmaxh = 5 # 15
maxh_vec = [0.2*0.83**n for n in range(nmaxh)] 

# print(maxh_vec)

full = {
    'min_res': [],
    'mean_res': [],
    'max_res': [],
    'calc_time': []
}

full_qep = {
    'min_res': [],
    'mean_res': [],
    'max_res': [],
    'calc_time': []
}

rb = {
    'min_res': [],
    'mean_res': [],
    'max_res': [],
    'offline_time': [],
    'online_time': []
}

rb_qep = {
    'min_res': [],
    'mean_res': [],
    'max_res': [],
    'offline_time': [],
    'online_time': []
}

dim_rb = []
dim_rb_qep = []

ndof = []

cnt = 0
for maxh in maxh_vec: 
    cnt += 1

    air = WorkPlane().RectangleC(1,1).Face()
    rod = WorkPlane().Circle(0,0,0.11).Face()
    mesh = CreateGeometry(air=air, rod=rod, max_h=maxh)

    mu, kappa = Permeability(freq=4.28, h0=0.16)
    eps = 15
    min_omega = 1e-4
    # max_omega = 0.625
    max_omega = 0.71
    nbands = 4

    dispGEP = DispersionGEP(mesh=mesh, mu=mu, kappa=kappa, eps=eps, order = order)
    dispQEP = DispersionQEP(mesh=mesh, mu=mu, kappa=kappa, eps=eps, order = order, param_omega=Parameter(1))

    print("\n({}/{}) ndof = {}".format(cnt, len(maxh_vec), dispGEP.fes.ndof))
    ndof.append(dispGEP.fes.ndof)

    print("\nGEP")
    values, times = dispGEP.CalculateBandStructure(nbands=nbands, nparam=nparam, nval=nval, th_res = th_res, min_omega=min_omega, max_omega = max_omega, useRB=False, plot = False)

    full['min_res'].append(np.min(values['res']))
    full['mean_res'].append(np.mean(values['res']))
    full['max_res'].append(np.max(values['res']))
    full['calc_time'].append(times['offline'])

    values, times = dispGEP.CalculateBandStructure(nbands=nbands, nparam=nparam, nval=nval, th_res = th_res, min_omega=min_omega, max_omega = max_omega, useRB=True, plot = False)

    rb['min_res'].append(np.min(values['res']))
    rb['mean_res'].append(np.mean(values['res']))
    rb['max_res'].append(np.max(values['res']))
    rb['offline_time'].append(times['offline'])
    rb['online_time'].append(times['online'])

    dim_rb.append(len(dispGEP.Qred))

    print("\nQEP")

    values, times, res = dispQEP.CalculateBandStructure(nparam=nparam, nval=nval, th_res = th_res, min_omega=min_omega, max_omega = max_omega, useRB=False, plot = False)

    full_qep['min_res'].append(np.min(res['min']))
    m = 0
    s = 0
    for i in range(len(values)):
        m += res['mean'][i]*len(values[i]['f'])
        s += len(values[i]['f'])
    full_qep['mean_res'].append(m/s)
    full_qep['max_res'].append(np.max(res['max']))
    full_qep['calc_time'].append(times['offline'])

    values, times, res = dispQEP.CalculateBandStructure(nparam=nparam, nval=nval, th_res = th_res, min_omega=min_omega, max_omega = max_omega, useRB=True, plot = False)

    rb_qep['min_res'].append(np.min(res['min']))
    m = 0
    s = 0
    for i in range(len(values)):
        m += res['mean'][i]*len(values[i]['f'])
        s += len(values[i]['f'])
    rb_qep['mean_res'].append(m/s)
    rb_qep['max_res'].append(np.max(res['max']))
    rb_qep['offline_time'].append(times['offline'])
    rb_qep['online_time'].append(times['online'])

    dim_rb_qep.append(len(dispQEP.Qred))

    # print(times)


pre = 'bandstructureI_RBvsFull_'

######### GEP #########

title = pre +'full_gep_res'
plt.figure(title)
plt.plot(ndof, full['min_res'], label='min residual')
plt.plot(ndof, full['mean_res'], label='mean residual')
plt.plot(ndof, full['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'full_gep_res_log'
plt.figure(title)
plt.plot(ndof, full['min_res'], label='min residual')
plt.plot(ndof, full['mean_res'], label='mean residual')
plt.plot(ndof, full['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass


title = pre +'rb_gep_res'
plt.figure(title)
plt.plot(ndof, rb['min_res'], label='min residual')
plt.plot(ndof, rb['mean_res'], label='mean residual')
plt.plot(ndof, rb['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass


title = pre +'rb_gep_res_log'
plt.figure(title)
plt.plot(ndof, rb['min_res'], label='min residual')
plt.plot(ndof, rb['mean_res'], label='mean residual')
plt.plot(ndof, rb['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'times_gep'
plt.figure(title)
plt.plot(ndof, full['calc_time'], label='without model order reduction')
plt.plot(ndof, rb['offline_time'], label='offline')
plt.plot(ndof, rb['online_time'], label='online')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
# plt.xscale('log')
plt.ylabel('time [s]')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'times_log_gep'
plt.figure(title)
plt.plot(ndof, full['calc_time'], label='without model order reduction')
plt.plot(ndof, rb['offline_time'], label='offline')
plt.plot(ndof, rb['online_time'], label='online')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
plt.xscale('log')
plt.ylabel('time [s]')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'dim_log_gep'
plt.figure(title)
plt.plot(ndof, dim_rb)
plt.xlim(min(ndof), max(ndof))
plt.xscale('log')
plt.ylabel('dimension of RB space')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'dim_log_gep'
plt.figure(title)
plt.plot(ndof, dim_rb)
plt.xlim(min(ndof), max(ndof))
plt.xscale('log')
plt.ylabel('dimension of RB space')
plt.xlabel('degrees of freedom')
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

print("dim rb gep:", dim_rb)
print("mean dim rb gep: ", np.mean(dim_rb))
print("std dim rb gep: ", np.std(dim_rb))


######### QEP #########

title = pre +'full_qep_res'
plt.figure(title)
plt.plot(ndof, full_qep['min_res'], label='min residual')
plt.plot(ndof, full_qep['mean_res'], label='mean residual')
plt.plot(ndof, full_qep['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass


title = pre +'full_qep_res_log'
plt.figure(title)
plt.plot(ndof, full_qep['min_res'], label='min residual')
plt.plot(ndof, full_qep['mean_res'], label='mean residual')
plt.plot(ndof, full_qep['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'rb_qep_res'
plt.figure(title)
plt.plot(ndof, rb_qep['min_res'], label='min residual')
plt.plot(ndof, rb_qep['mean_res'], label='mean residual')
plt.plot(ndof, rb_qep['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'rb_qep_res_log'
plt.figure(title)
plt.plot(ndof, rb_qep['min_res'], label='min residual')
plt.plot(ndof, rb_qep['mean_res'], label='mean residual')
plt.plot(ndof, rb_qep['max_res'], label='max residual')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'times_qep'
plt.figure(title)
plt.plot(ndof, full_qep['calc_time'], label='without model order reduction')
plt.plot(ndof, rb_qep['offline_time'], label='offline')
plt.plot(ndof, rb_qep['online_time'], label='online')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
# plt.xscale('log')
plt.ylabel('time [s]')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass


title = pre +'times_log_qep'
plt.figure(title)
plt.plot(ndof, full_qep['calc_time'], label='without model order reduction')
plt.plot(ndof, rb_qep['offline_time'], label='offline')
plt.plot(ndof, rb_qep['online_time'], label='online')
plt.xlim(min(ndof), max(ndof))
plt.yscale('log')
plt.xscale('log')
plt.ylabel('time [s]')
plt.xlabel('degrees of freedom')
plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass

title = pre +'dim_log_qep'
plt.figure(title)
plt.plot(ndof, dim_rb_qep)
plt.xlim(min(ndof), max(ndof))
plt.xscale('log')
plt.ylabel('dimension of RB space')
plt.xlabel('degrees of freedom')
# plt.legend()
plt.tight_layout()
try:
    plt.savefig('../output/'+title+'.png')
except:
    pass


print("dim rb qep:", dim_rb_qep)
print("mean dim rb qep: ", np.mean(dim_rb_qep))
print("std dim rb qep: ", np.std(dim_rb_qep))




# OUTPUT:

[36, 38, 34, 34, 36, 36, 36, 36, 34, 32, 30, 30, 28, 28, 26] # gep
[34, 36, 34, 36, 34, 32, 32, 36, 32, 26, 32, 24, 28, 24, 22] # qep

plt.show()
