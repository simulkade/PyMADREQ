import pymadreq.coreflood as cf
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d

def frac_flow_wf(fluids: cf.Fluids, 
                 rel_perm: cf.RelativePermeability,
                 core_plug: cf.CorePlug,
                 IC: cf.InitialConditions,
                 ut=1e-5, sw_inj=1.0, pv_inj=5.0):
    sw0 = IC.sw
    L = core_plug.core_length
    fw = lambda sw: ((rel_perm.krw(sw)/fluids.water_viscosity)/(rel_perm.krw(sw)/fluids.water_viscosity+rel_perm.kro(sw)/fluids.oil_viscosity))
    dfw = lambda sw: ((rel_perm.dkrwdsw(sw)/fluids.water_viscosity*(rel_perm.krw(sw)/fluids.water_viscosity+rel_perm.kro(sw)/fluids.oil_viscosity)- \
    (rel_perm.dkrwdsw(sw)/fluids.water_viscosity+rel_perm.dkrodsw(sw)/fluids.oil_viscosity)*rel_perm.krw(sw)/fluids.water_viscosity)/ \
    (rel_perm.kro(sw)/fluids.oil_viscosity+rel_perm.krw(sw)/fluids.water_viscosity)**2)
    # eps1=1e-3
    # dfw_num(sw) = (fw(sw+eps1)-fw(sw-eps1))/(2*eps1)
    # ftot(sw)=kro_new(sw)/muo+krw_new(sw)/muw

    # solve the nl equation to find the shock front saturation
    eps = 1e-15
    f_shock = lambda sw: (dfw(sw)-(fw(sw)-fw(sw0))/(sw-sw0))
    sw_tmp = np.linspace(rel_perm.swc+eps, 1-rel_perm.sor-eps, 1000)
    sw_shock = sw_tmp[np.abs(f_shock(sw_tmp)).argmin()]
    try:
        if f_shock(rel_perm.swc+eps)*f_shock(1-rel_perm.sor-eps)>0:
            sw_shock = opt.newton(f_shock, sw_shock) #  ftol = 1e-10, xtol = 1e-7)
        else:
            sw_shock = opt.brentq(f_shock, rel_perm.swc+eps, 1-rel_perm.sor-eps)
    except:
        print('shock front saturation is estimated: $sw_shock, error is $(f_shock(sw_shock))')
    # s = np.linspace(0.0, 1.0, 100)
    s1 = np.linspace(sw_inj, sw_shock-eps, 1000)
    xt_s1 = ut/core_plug.porosity*dfw(s1)
    xt_shock = ut/core_plug.porosity*dfw(sw_shock)
    xt_prf=np.concatenate((xt_s1, xt_shock, xt_shock+eps, 2*xt_shock))
    sw_prf=np.concatenate((s1, [sw_shock, sw0, sw0]))
    # println(xt_prf)
    # print(sw_shock)
    # Go through the data first
    i=1
    xt_prf, indices = np.unique(xt_prf, return_index = True)
    sw_prf = sw_prf[indices]
    
    # xt_prf_slim = unique(xt_prf)
    # sw_prf = sw_prf[indexin(xt_prf_slim, xt_prf)]
    # xt_prf = xt_prf_slim

    # find the injection pressure history
    x = np.linspace(0,L,1000)
    # sw_int = Spline1D([xt_prf; L/eps()], [sw_prf; sw0], k=1)
    # println(xt_prf)
    # println(sw_prf)
    sw_int = interp1d(xt_prf, sw_prf, kind='linear', fill_value='extrapolate')
    t_inj=pv_inj*core_plug.porosity*L/ut
    t = np.linspace(0.0,t_inj, 200) # [s] time
    p_inj = np.zeros(t.size)
    R_oil= np.zeros(t.size)
    p_inj[0]=np.trapz(ut/(core_plug.permeability*(rel_perm.kro(sw0*np.ones(np.size(x)))/fluids.oil_viscosity+rel_perm.krw(sw0*np.ones(np.size(x)))/fluids.water_viscosity)), x=x)
    for i in range(1,t.size):
        xt = x/t[i]
        p_inj[i] = np.trapz(ut/(core_plug.permeability*(rel_perm.kro(sw_int(xt))/fluids.oil_viscosity+rel_perm.krw(sw_int(xt))/fluids.water_viscosity)), x=x)
        R_oil[i]=1.0-np.trapz(1.0-sw_int(xt), x = x/L)/(1-sw0)

    # Return the results
    return (xt_shock, sw_shock, xt_prf, sw_prf, t, p_inj, R_oil)
