import json
import os
import sys
from pyfvtool import *
import pymadreq.coreflood as cf

def read_json(json_file):
    """
    Read a json file and return a dictionary. See sample json files in the examples folder.
    """
    try:
        with open(json_file) as f:
            data = json.load(f)
    except:
        print("Error: Could not read json file")
        sys.exit(1)
    return data

def createFluids(fluids_dict: dict):
    """
    Creates a Fluids object from a dictionary of fluid properties.
    """
    return cf.Fluids(mu_water=fluids_dict["water"]["viscosity"], mu_oil=fluids_dict["oil"]["viscosity"])

def createCapillaryPressurePiecewise(capillary_pressure_dict: dict, rel_perm_dict: dict):
    """
    Creates a CapillaryPressurePiecewise object from a dictionary of capillary pressure properties.
    """
    return cf.CapillaryPressurePiecewise(sw_pc0=capillary_pressure_dict["sw_pc0"], pce=capillary_pressure_dict["pcw_entry"], 
                                      sorting_factor=capillary_pressure_dict["labda_w"],
                                      pc_min=capillary_pressure_dict["pc_min"], pc_max=capillary_pressure_dict["pc_max"], 
                                      pc_lm=capillary_pressure_dict["pc_lm"], pc_hm=capillary_pressure_dict["pc_hm"], 
                                      swc=rel_perm_dict["swc"], sor=rel_perm_dict["sor"], 
                                      extrap_factor=capillary_pressure_dict["extrap_factor"], 
                                      curve_factor_l=capillary_pressure_dict["curve_low"], 
                                      curve_factor_h=capillary_pressure_dict["curve_high"])

def createRelativePermeabilityCorey(rel_perm_dict: dict):
    """
    Creates a RelativePermeability object from a dictionary of relative permeability properties.
    """
    return cf.RelativePermeability(swc=rel_perm_dict["swc"], sor=rel_perm_dict["sor"], 
                                kro0=rel_perm_dict["kro0"], no=rel_perm_dict["no"], 
                                krw0=rel_perm_dict["krw0"], nw=rel_perm_dict["nw"])

def createCapillaryPressureCorey(capillary_pressure_dict: dict, rel_perm_dict: dict):
    """
    Creates a CapillaryPressure object from a dictionary of Corey-type capillary pressure properties.
    """
    return cf.CapillaryPressureBrooksCorey(swc=rel_perm_dict["swc"], sor=rel_perm_dict["sor"], 
                                        pce_w=capillary_pressure_dict["pcw_entry"], pce_o=capillary_pressure_dict["pco_entry"], 
                                        labda_w=capillary_pressure_dict["labda_w"], labda_o=capillary_pressure_dict["labda_o"], 
                                        pc_max_w=capillary_pressure_dict["pc_max_w"], pc_max_o=capillary_pressure_dict["pc_max_o"])

def createCore(plug_dict: dict):
    """
    Creates a CorePlug object from a dictionary of core properties.
    """
    return cf.CorePlug(porosity=plug_dict["porosity"], permeability=plug_dict["permeability"], 
                    diameter=plug_dict["diameter"], core_length=plug_dict["length"])

def createIC(ic_dict: dict):
    """
    Creates a InitialConditions object from a dictionary of initial conditions properties.
    """
    return cf.InitialConditions(pressure=ic_dict["pressure"], 
                                saturation=ic_dict["water_saturation"],
                                temperature=ic_dict["temperature"],
                                salinity=ic_dict["salinity"])

def createOperationalCondition(op_dict: dict):
    """
    Creates a Operation object from a dictionary of operating and boundary conditions
    """
    return cf.OperationalConditions()

def createFloodingDomain(data: dict):
    return createMesh1D(data["flooding"]["Nx"], data["core"]["length"])

def createImbibitionDomain(data: dict):
    return createMeshCylindrical2D(data["imbibition"]["Nx"], data["imbibition"]["Ny"], 
                        data["core"]["diameter"]/2, data["core"]["length"])

def read_transport_functions(data: dict, pc_model: str = 'piecewise'):
    """
    Create relative permeability and capillary pressure functions using an input dictionary
    read from an input json file
    """
    rel_perm_ww = createRelativePermeabilityCorey(data['rel_perm']['water_wet']['corey'])
    rel_perm_ow = createRelativePermeabilityCorey(data['rel_perm']['oil_wet']['corey'])
    if pc_model == 'piecewise':
        cap_pres_ww = createCapillaryPressurePiecewise(data['pc']['water_wet'][pc_model])
        cap_pres_ow = createCapillaryPressurePiecewise(data['pc']['oil_wet'][pc_model])
    elif pc_model == 'corey':
        cap_pres_ww = createCapillaryPressureCorey(data['pc']['water_wet'][pc_model])
        cap_pres_ow = createCapillaryPressureCorey(data['pc']['oil_wet'][pc_model])

    return rel_perm_ww, rel_perm_ow, cap_pres_ww, cap_pres_ow

def read_fluids(data: dict):
    """
    Read fluid properties from an input dictionary read from an input json file
    """
    return cf.Fluids(mu_water= data['phase']['water']['viscosity'], 
                      rho_water= data['phase']['water']['density'], 
                      mu_oil= data['phase']['oil']['viscosity'],
                      rho_oil= data['phase']['oil']['density'])

def read_core_properties(data: dict):
    """
    Read core properties from an input dictionary read from an input json file
    """
    return cf.createCore(data)

def read_flooding(data: dict):
    """
    Read flooding properties from an input dictionary read from an input json file
    """
    pass