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

def createFloodingCondition(op_dict: dict):
    """
    Creates a Operation object from a dictionary of operating and boundary conditions
    """
    return cf.FloodingConditions(injection_rate_ml_min=op_dict["injection_rate_ml_min"], 
                                    injection_pressure=op_dict["pressure"],
                                    production_pressure=op_dict["back_pressure"],
                                    injection_sw=op_dict["water_saturation"],
                                    active_rate=True)

# All functions that begin with read_ are used to read input data from the dictionary coming from a json file, as opposed to the create_ functions
# that are used to create objects from specific dictionaries
def read_flooding_domain(data: dict):
    return createMesh1D(data["flooding"]["Nx"], data["core"]["length"])

def read_imbibition_domain(data: dict):
    return createMeshCylindrical2D(data["imbibition"]["Nx"], data["imbibition"]["Ny"], 
                        data["core"]["diameter"]/2, data["core"]["length"])

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

def read_transport_functions(data: dict, pc_model: str = 'piecewise'):
    """
    Create relative permeability and capillary pressure functions using an input dictionary
    read from an input json file
    """
    rel_perm_ww = createRelativePermeabilityCorey(data['rel_perm']['water_wet']['corey'])
    rel_perm_ow = createRelativePermeabilityCorey(data['rel_perm']['oil_wet']['corey'])
    if pc_model == 'piecewise':
        cap_pres_ww = createCapillaryPressurePiecewise(data['pc']['water_wet'][pc_model], 
                                                       data['rel_perm']['water_wet']['corey'])
        cap_pres_ow = createCapillaryPressurePiecewise(data['pc']['oil_wet'][pc_model],
                                                       data['rel_perm']['oil_wet']['corey'])
    elif pc_model == 'corey':
        cap_pres_ww = createCapillaryPressureCorey(data['pc']['water_wet'][pc_model],
                                                   data['rel_perm']['water_wet']['corey'])
        cap_pres_ow = createCapillaryPressureCorey(data['pc']['oil_wet'][pc_model],
                                                   data['rel_perm']['oil_wet']['corey'])

    return rel_perm_ww, rel_perm_ow, cap_pres_ww, cap_pres_ow

def read_fluids(data: dict):
    """
    Read fluid properties from an input dictionary read from an input json file
    """
    return cf.Fluids(mu_water= data['phase']['water']['viscosity'], 
                      rho_water= data['phase']['water']['density'], 
                      mu_oil= data['phase']['oil']['viscosity'],
                      rho_oil= data['phase']['oil']['density'])

def read_initial_conditions(data: dict):
    """
    Read initial conditions from an input dictionary read from an input json file
    """
    return cf.InitialConditions(pressure=data["IC"]["pressure"], 
                                water_saturation=data["IC"]["water_saturation"],
                                temperature=data["IC"]["temperature"],
                                salinity=data["IC"]["salinity"])

def read_core_properties(data: dict):
    """
    Read core properties from an input dictionary read from an input json file
    """
    return cf.CorePlug(porosity=data["core"]["porosity"], permeability=data["core"]["permeability"], 
                    diameter=data["core"]["diameter"], core_length=data["core"]["length"])

def read_numerical_settings(data: dict):
    return cf.NumericalSettings(time_step=data["numeric"]["dt"],
                                eps_sw = data["numeric"]["eps_sw"],
                                eps_p = data["numeric"]["eps_p"],
                                dp_allowed=data["numeric"]["allowed_dp"],
                                dsw_allowed=data["numeric"]["allowed_dsw"],
                                simulation_time=data["numeric"]["end_time"],
                                )

def read_flooding_settings(data: dict):
    """
    Read flooding properties from an input dictionary read from an input json file
    """
    return cf.FloodingConditions(injection_rate_ml_min=data["flooding"]["injection_rate_ml_min"],
                                 injection_pressure=data["flooding"]["pressure"],
                                 production_pressure=data["flooding"]["back_pressure"],
                                 injection_sw=data["flooding"]["water_saturation"],
                                 active_rate=True)

def read_imbibition_settings(data: dict):
    """
    Read imbibition properties from an input dictionary read from an input json dictionary
    """
    pass    