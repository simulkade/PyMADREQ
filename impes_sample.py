from importlib import reload
from pymadreq import *
import pymadreq.coreflood as cf
import pymadreq.IO as IO
import matplotlib.pyplot as plt
import numpy as np
reload(cf)
reload(IO)

from pyfvtool import *

# read the input file
data = IO.read_json("examples\sample.json")
rel_perm_ww, rel_perm_ow, pc_ww, pc_ow = IO.read_transport_functions(data)
m = IO.createFloodingDomain(data)

# define the physical parametrs
SF0 = 0.5 # closer to 1 is water-wet
SF_cell = createCellVariable(m, SF0)
SF=createFaceVariable(m, SF0) # 1 is water wet, 0 is oil wet
# define functions for face variables saturation and wettability indicator
krw = @(sw, sf)(krw_ww(sw).*sf+krw_ow(sw).*(1-sf))
kro = @(sw, sf)(kro_ww(sw).*sf+kro_ow(sw).*(1-sf))
dkrwdsw = @(sw, sf)(dkrw_ww(sw).*sf+dkrw_ow(sw).*(1-sf))
dkrodsw = @(sw, sf)(dkro_ww(sw).*sf+dkro_ow(sw).*(1-sf))
pc_imb = @(sw, sf)(pc_imb_ww(sw).*sf+pc_imb_ow(sw).*(1-sf))
dpc_imb = @(sw, sf)(dpc_imb_ww(sw).*sf+dpc_imb_ow(sw).*(1-sf))

# core
k= perm_field
phi= poros_field
lw = geometricMean(k)./arithmeticMean(mu_water)
lo = geometricMean(k)./arithmeticMean(mu_oil)
# A_core = 1
# Initial and boundary conditions
p0 = flooding_struct.initial_p # [Pa] pressure
p_back = flooding_struct.back_pressure # [Pa] pressure
u_inj = flooding_struct.injection_velocity # [m/s]
sw0 = flooding_struct.initial_sw # initial saturation
sw_pc0 = fzero(@(sw)(pc_imb(sw, SF0)), [0.01, 0.99]) # saturation at which pc=0
BCp = createBC(m) # Neumann BC for pressure
BCs = createBC(m) # Neumann BC for saturation
BCp.right.a(:)=0.0 BCp.right.b(:)=1.0 BCp.right.c(:)=p_back
BCp.left.a(:)=1.0 BCp.left.b(:)=0.0 BCp.left.c(:)=-u_inj/lw.xvalue(1)
# BCs.right.a(:)=0 BCs.right.b(:)=1 BCs.right.c(:)=sw_pc0
BCs.left.a(:)=0 BCs.left.b(:)=1 BCs.left.c(:)=1.0

# initial conditions
sw_old = createCellVariable(m, sw0, BCs)
p_old = createCellVariable(m, p0, BCp)
p = p_old
sw = sw_old
uw = -gradientTerm(p_old) # an estimation of the water velocity
oil_init=domainInt(1-sw_old)
rec_fact=0
t_day=0
t = 0

# Concentrations
# How many components? Here, I assume all the relevant components are added
# by the user in the initial condition (including a small value of 1e-16
# for components with zero concentration)


# define the time step and solver properties
dt=1
dt0=dt
t_end = flooding_struct.injection_end_time # [s] final simulation time
eps_p = flooding_struct.eps_p # pressure accuracy
eps_sw = flooding_struct.eps_sw # saturation accuracy
dsw_alwd= flooding_struct.allowed_dsw
dp_alwd= flooding_struct.allowed_dp # Pa


# start the main loop
reverseStr='' # show the progress
t_end1 = t_end*0.8
t_end2 = 1.2*t_end
while (t<t_end2)
# for i=1:5
    error_p = 1e5
    error_sw = 1e5
    # Implicit loop
#     while ((error_p>eps_p) || (error_sw>eps_sw))
    while(1) # loop condition is checked inside
        # calculate parameters
        pgrad = gradientTerm(p)
#         pcgrad=gradientTerm(pc(sw))
        sw_face = upwindMean(sw, -pgrad) # average value of water saturation
        sw_grad=gradientTerm(sw)
        sw_ave=arithmeticMean(sw)
#         pcval=pc(sw)
#         pcgrad=gradientTerm(pcval)
#         pcgrad=funceval(dpc, sw_ave, pce, swc, sor, teta).*sw_grad
#         pcgrad=funceval(dpc_imb, sw_ave, SF).*sw_grad
        pc_cell = celleval(pc_imb, BC2GhostCells(sw), SF_cell)
        pcgrad = gradientTermFixedBC(pc_cell)
#         pcgrad=funceval(dpc_imb, sw_face, SF).*sw_grad
        # solve for pressure at known Sw
        labdao = lo.*funceval(kro, sw_face, SF)
        labdaw = lw.*funceval(krw, sw_face, SF)
        labda = labdao+labdaw
        # compute [Jacobian] matrices
        Mdiffp1 = diffusionTerm(-labda)
        RHSpc1=divergenceTerm(labdao.*pcgrad)
        if t>t_end1
            BCp.left.a(:)=1.0 BCp.left.b(:)=0.0 BCp.left.c(:)=-5*u_inj/lw.xvalue(1)
        end
        [Mbcp, RHSbcp] = boundaryCondition(BCp)
        RHS1 = RHSpc1+RHSbcp # with capillary
        p_new=solvePDE(m, Mdiffp1+Mbcp, RHS1)
        
        # solve for Sw
        pgrad = gradientTerm(p_new)
        uw=-labdaw.*pgrad
        [Mbcsw, RHSbcsw] = boundaryCondition(BCs)
        RHS_sw=-divergenceTerm(uw)
        sw_new=solveExplicitPDE(sw_old, dt, RHS_sw, BCs, phi)

        error_p = max(abs((p_new.value(:)-p.value(:))./p_new.value(:)))
        error_sw = max(abs(sw_new.value(2:end-1)-sw.value(2:end-1)))
        # dt_new=dt*min(dp_alwd/error_p, dsw_alwd/error_sw) # p fluctuates
        # way too much sometimes
        dt_new=dt*dsw_alwd/error_sw
        # assign new values of p and sw
        if error_sw>dsw_alwd
            dt=dt*(dsw_alwd/error_sw)
        else
            t=t+dt
            p = p_new
            sw = sw_new
            p_old = p
            sw_old = sw
            
            dt=min(dt*(dsw_alwd/error_sw), 10*dt)
            break
        end
    end
    
    # Calculate the new concentration profile after advection-diffusion
    
    # calculate the new concentrations after chemical reactions
    
    # update rel-perm and pc curves

    # new boundary
    if sw_ave.xvalue(end)>=sw_pc0
        BCs.right.a(:)=0 BCs.right.b(:)=1 BCs.right.c(:)=sw_pc0
    end
    # calculate recovery factor
    rec_fact=[rec_fact (oil_init-domainInt(1-sw))/oil_init]
    t_day=[t_day t]

    percentDone = 100 * t / t_end
    msg = sprintf('Percent done: #3.1f', percentDone)
    fprintf([reverseStr, msg])
    reverseStr = repmat(sprintf('\b'), 1, length(msg))
end
figure(2) plot(t_day/3600/24, rec_fact)
xlabel('time [day]')
ylabel('recovery factor')
title([num2str(t/3600/24) ' day']) drawnow