from pyfvtool import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d, pchip
from scipy.optimize import root
from tqdm import tqdm


def eps():
    return np.finfo(float).eps


class CorePlug:
    """
    This class represents a core plug properties and functions.
    """

    def __init__(
        self, porosity=0.22, permeability=1e-12, diameter=2.54, core_length=6.0
    ) -> None:
        self.porosity = porosity
        self.permeability = permeability
        self.diameter = diameter
        self.core_length = core_length
        self.pore_volume = porosity * core_length * np.pi * (diameter / 2) ** 2
        self.cross_sectional_area = np.pi * (diameter / 2) ** 2


class CapillaryPressure:
    def __init__(self, swc=0.1, sor=0.1):
        self.swc = swc
        self.sor = sor

    def visualize(self):
        plt.figure()
        sw = np.linspace(0.0, 1.0, 100)
        plt.plot(sw, self.pc_imb(sw))
        plt.plot(sw, self.pc_drain(sw))
        if type(self) == CapillaryPressurePiecewise:
            plt.plot(self.imb_points[:, 0], self.imb_points[:, 1], "o")


class CapillaryPressurePiecewise(CapillaryPressure):
    def __init__(
        self,
        sw_pc0=0.6,
        pce=1e5,
        sorting_factor=2.0,
        pc_min=-1e6,
        pc_max=5e6,
        pc_lm=-5e4,
        pc_hm=7e4,
        swc=0.15,
        sor=0.2,
        extrap_factor=200.0,
        curve_factor_l=5.0,
        curve_factor_h=10.0,
    ):
        """
        This class defines a piecewise capillary pressure curve for imbibition.
        The curve is defined by fitting a monotonic cubic spline to the following points:
        pc = 0 at sw = sw_pc0
        pc = pc_min at sw = 1.0-Sor
        pc = pc_max at sw = Swc
        pc = pc_lm at sw_pc0 < Sw < 1-Sor
        pc = pc_hm at Swc < Sw < sw_pc0
        Subject to:
        pc_min<pc_lm<0<pc_hm<pc_max
        curve_factor_l>1.0 (recommended value of > 2.0)
        curve_factor_h>1.0 (recommended value of > 2.0)
        """
        super().__init__(swc=swc, sor=sor)
        self.sw_pc0 = sw_pc0
        self.pc_min = pc_min
        self.pc_max = pc_max

        sw_hm = np.mean([swc, sw_pc0])
        sw_lm = np.mean([1 - sor, sw_pc0])
        sw_curve_h = swc + (sw_hm - swc) / curve_factor_h
        sw_curve_l = (1 - sor) - (1 - sor - sw_lm) / curve_factor_l

        pc_curve_h = interp1d(
            [sw_hm, sw_pc0], [pc_hm, 0], kind="linear", fill_value="extrapolate"
        )(sw_curve_h)
        pc_curve_h = pc_curve_h + (pc_max - pc_curve_h) / curve_factor_h

        pc_curve_l = interp1d(
            [sw_pc0, sw_lm], [0, pc_lm], kind="linear", fill_value="extrapolate"
        )(sw_curve_l)
        pc_curve_l = pc_curve_l - (pc_curve_l - pc_min) / curve_factor_l

        pc_data = np.array(
            [
                [0.0, extrap_factor * pc_max],
                [swc, pc_max],
                [sw_curve_h, pc_curve_h],
                [sw_hm, pc_hm],
                [sw_pc0, 0.0],
                [sw_lm, pc_lm],
                [sw_curve_l, pc_curve_l],
                [1 - sor, pc_min],
                [1.0, extrap_factor * pc_min],
            ]
        )
        # print(pc_data)
        pc_pp = pchip(pc_data[:, 0], pc_data[:, 1])
        # pc = lambda sw: pc_pp(sw)
        pc_der_pp = pc_pp.derivative(nu=1)
        # pcder = lambda sw: pc_der_pp(sw)
        self.imb_points = pc_data
        self.pc_imb = pc_pp
        self.dpc_dsw_imb = pc_der_pp

        # Define drainage curve
        self.labda = sorting_factor  # -np.log((swc)/(1-swc))/(np.log(pc_max/pce)) until I find a better way
        pc_corey = CapillaryPressureBrooksCorey(
            swc=swc, sor=sor, pce_w=pce, labda_w=self.labda, pc_max_w=pc_max
        )
        # self.labda = labda
        # print(self.labda)
        # print("sw0_w", pc_corey.sw0_w)
        pc_drain_data = np.array(
            [
                [0.0, extrap_factor * pc_max],
                [swc, pc_max],
                # [pc_corey.sw0, pc_corey.pc_drain(pc_corey.sw0)],
                # [np.mean([pc_corey.sw0_w, sw_hm]), pc_corey.pc_drain(np.mean([pc_corey.sw0_w, sw_hm]))],
                [sw_hm, pc_corey.pc_drain(sw_hm)],
                [sw_pc0, pc_corey.pc_drain(np.array(sw_pc0))],
                [1 - sor, pc_corey.pc_drain(np.array(1 - sor))],
                [1.0, pce],
            ]
        )
        # print(pc_drain_data)
        self.pc_drain = pchip(pc_drain_data[:, 0], pc_drain_data[:, 1])
        self.dpc_dsw_drain = self.pc_drain.derivative(nu=1)


class CapillaryPressureBrooksCorey(CapillaryPressure):
    def __init__(
        self,
        swc=0.1,
        sor=0.1,
        pce_w=1e5,
        pce_o=1e5,
        labda_w=2.0,
        labda_o=2.0,
        pc_max_w=5e6,
        pc_max_o=5e6,
    ):
        super().__init__(swc=swc, sor=sor)
        self.pce_w = pce_w
        self.pce_o = pce_o
        self.labda_o = labda_o
        self.labda_w = labda_w
        self.pc_max_w = pc_max_w
        self.pc_max_o = pc_max_o
        self.sw0_w = swc + (
            1
            - labda_w * np.log(pc_max_w / pce_w)
            + np.sqrt(
                (-1 + labda_w * np.log(pc_max_w / pce_w)) ** 2 + 4 * swc / (1 - swc)
            )
        ) / 2 * (1 - swc)
        self.sw0_o = sor + (
            1
            - labda_o * np.log(pc_max_o / pce_o)
            + np.sqrt(
                (-1 + labda_o * np.log(pc_max_o / pce_o)) ** 2 + 4 * sor / (1 - sor)
            )
        ) / 2 * (1 - sor)
        self.pcs_w = pce_w * ((self.sw0_w - swc) / (1 - swc)) ** (-1.0 / labda_w)
        self.pcs_o = pce_o * ((self.sw0_o - sor) / (1 - sor)) ** (-1.0 / labda_o)

    # function res=pc_imb(sw, pce_w, pce_o, swc, sor, labda_w, labda_o, pc_max_w, pc_max_o)
    #   pc1=pc_drain(sw, pce_w, swc, labda_w, pc_max_w);
    #   pc2=pc_drain(1-sw, pce_o, sor, labda_o, pc_max_o);
    #   res=pc1-pc2;
    # end
    def _pc(self, sw, pce, labda, pc_max, sw0, pcs, swc):
        res = np.zeros_like(sw)
        cond1 = np.logical_and(0.0 <= sw, sw <= sw0)
        res[cond1] = np.exp(
            (np.log(pcs) - np.log(pc_max)) / sw0 * (sw[cond1] - sw0) + np.log(pcs)
        )
        res[sw > sw0] = pce * ((sw[sw > sw0] - swc + eps()) / (1 - swc)) ** (
            -1.0 / labda
        )
        res[sw < 0.0] = pc_max
        return res

    def pc_drain(self, sw):
        return self._pc(
            sw,
            self.pce_w,
            self.labda_w,
            self.pc_max_w,
            self.sw0_w,
            self.pcs_w,
            self.swc,
        )

    def pc_imb(self, sw):
        return self.pc_drain(sw) - self._pc(
            1 - sw,
            self.pce_o,
            self.labda_o,
            self.pc_max_o,
            self.sw0_o,
            self.pcs_o,
            self.sor,
        )

    def dpc_dsw_drain(self, sw):
        res = np.zeros_like(sw)
        cond1 = np.logical_and(0.0 <= sw, sw < self.sw0)
        res[cond1] = (
            (np.log(self.pcs) - np.log(self.pc_max))
            / self.sw0
            * np.exp(
                (np.log(self.pcs) - np.log(self.pc_max))
                / self.sw0
                * (sw[cond1] - self.sw0)
            )
        )
        res[sw >= self.sw0] = (
            -self.pce
            / self.labda
            * ((sw[sw >= self.sw0] - self.swc + eps()) / (1 - self.swc))
            ** (-1.0 / self.labda - 1.0)
        )
        res[sw <= 0.0] = 0.0
        return res


class Fluids:
    """
    This class represents a model for calculating fluid properties.

    Attributes:
        water_viscosity (float): Viscosity of water. Default is 0.001 Pa.s.
        oil_viscosity (float): Viscosity of oil. Default is 0.003 Pa.s.
    """

    def __init__(self, mu_water=0.001, mu_oil=0.003, rho_water=1000.0, rho_oil=800.0):
        self.water_viscosity = mu_water
        self.oil_viscosity = mu_oil
        self.water_density = rho_water  # kg/m3
        self.oil_density = rho_oil  # kg/m3


class RelativePermeability:
    """
    This class represents a model for calculating relative permeability values
    based on water saturation (sw) using the Corey-Brooks model.

    Attributes:
        swc (float): Connate water saturation. Default is 0.1.
        sor (float): Residual oil saturation. Default is 0.05.
        kro0 (float): Oil relative permeability at 100% water saturation. Default is 0.9.
        no (float): Corey exponent for oil relative permeability. Default is 2.0.
        krw0 (float): Water relative permeability at 100% water saturation. Default is 0.4.
        nw (float): Corey exponent for water relative permeability. Default is 2.0.
    """

    def __init__(self, swc=0.1, sor=0.05, kro0=0.9, no=2.0, krw0=0.4, nw=2.0):
        """
        Initializes a new instance of the RelativePermeability class.

        Args:
            swc (float, optional): Connate water saturation. Default is 0.1.
            sor (float, optional): Residual oil saturation. Default is 0.05.
            kro0 (float, optional): Oil relative permeability at 100% water saturation. Default is 0.9.
            no (float, optional): Corey exponent for oil relative permeability. Default is 2.0.
            krw0 (float, optional): Water relative permeability at 100% water saturation. Default is 0.4.
            nw (float, optional): Corey exponent for water relative permeability. Default is 2.0.
        """
        self.kro0 = kro0
        self.krw0 = krw0
        self.no = no
        self.nw = nw
        self.swc = swc
        self.sor = sor

    def kro(self, sw: np.ndarray) -> np.ndarray:
        """
        Calculates the oil relative permeability based on the water saturation.

        Args:
            sw (float): Water saturation.

        Returns:
            float: Oil relative permeability.
        """
        kro0 = self.kro0
        sor = self.sor
        swc = self.swc
        no = self.no
        if isinstance(sw, float):
            sw = np.array([sw])
        res = np.zeros_like(sw)
        cond1 = np.logical_and(swc <= sw, sw <= 1 - sor)
        res[cond1] = kro0 * ((1 - sw[cond1] - sor) / (1 - sor - swc)) ** no
        cond2 = np.logical_and(0.0 < sw, sw < swc)
        res[cond2] = 1 + (kro0 - 1) / swc * sw[cond2]
        res[sw > 1 - sor] = 0.0
        res[sw <= 0.0] = 1.0
        if isinstance(sw, float):
            return res[0]
        else:
            return res

    def krw(self, sw: np.ndarray) -> np.ndarray:
        """
        Calculates the water relative permeability based on the water saturation.

        Args:
            sw (float): Water saturation.

        Returns:
            float: Water relative permeability.
        """
        krw0, sor, swc, nw = self.krw0, self.sor, self.swc, self.nw
        if isinstance(sw, float):
            sw = np.array([sw])
        res = np.zeros_like(sw)
        cond1 = np.logical_and(swc <= sw, sw <= 1 - sor)
        res[cond1] = krw0 * ((sw[cond1] - swc) / (1 - sor - swc)) ** nw
        cond2 = np.logical_and(1 - sor < sw, sw < 1.0)
        res[cond2] = -(1 - krw0) / sor * (1.0 - sw[cond2]) + 1.0
        res[sw <= swc] = 0.0
        res[sw >= 1.0] = 1.0
        if isinstance(sw, float):
            return res[0]
        else:
            return res

    def dkrodsw(self, sw):
        """
        Calculates the derivative of oil relative permeability with respect to water saturation.

        Args:
            sw (float): Water saturation.

        Returns:
            float: Derivative of oil relative permeability with respect to water saturation.
        """
        kro0, sor, swc, no = self.kro0, self.sor, self.swc, self.no
        if isinstance(sw, float):
            sw = np.array([sw])
        res = np.zeros_like(sw)
        cond1 = np.logical_and(swc <= sw, sw <= 1 - sor)
        res[cond1] = (
            -no
            * kro0
            / (1 - sor - swc)
            * ((1 - sw[cond1] - sor) / (1 - sor - swc)) ** (no - 1)
        )
        cond2 = np.logical_and(0.0 < sw, sw < swc)
        res[cond2] = (kro0 - 1) / swc
        res[sw > 1 - sor] = 0.0
        res[sw <= 0.0] = 0.0
        if isinstance(sw, float):
            return res[0]
        else:
            return res

    def dkrwdsw(self, sw):
        """
        Calculates the derivative of water relative permeability with respect to water saturation.

        Args:
            sw (float): Water saturation.

        Returns:
            float: Derivative of water relative permeability with respect to water saturation.
        """
        krw0, sor, swc, nw = self.krw0, self.sor, self.swc, self.nw
        if isinstance(sw, float):
            sw = np.array([sw])
        res = np.zeros_like(sw)
        cond1 = np.logical_and(swc <= sw, sw <= 1 - sor)
        res[cond1] = (
            nw
            * krw0
            / (1 - sor - swc)
            * ((sw[cond1] - swc) / (1 - sor - swc)) ** (nw - 1)
        )
        cond2 = np.logical_and(1 - sor < sw, sw < 1.0)
        res[cond2] = (1 - krw0) / sor
        res[sw < swc] = 0.0
        res[sw >= 1.0] = 0.0
        if isinstance(sw, float):
            return res[0]
        else:
            return res

    def visualize(self):
        """
        Visualizes the relative permeability curves for water and oil.

        Requires matplotlib.pyplot to be imported as plt.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure()
        sw_ = np.linspace(0.0, 1.0, 50)
        plt.plot(sw_, self.krw(sw_), label="Water")
        plt.plot(sw_, self.kro(sw_), label="Oil")
        plt.xlabel("Water saturation")
        plt.ylabel("Relative permeability")
        plt.legend()


class Reservoir:
    def __init__(
        self,
        rel_perm: RelativePermeability,
        fluids: Fluids,
        core: CorePlug,
        sw_init=0.2,
        pressure_init=100e5,
    ):
        self.porosity = core.porosity
        self.permeability = core.permeability
        self.rel_perm = rel_perm
        self.fluids = fluids
        self.initial_sw = np.maximum(rel_perm.swc, sw_init)
        self.initial_p = pressure_init


class InitialConditions:
    """
    initial conditions for the core. This includes the initial pressure, saturation, temperature, and salinity
    """

    def __init__(
        self, water_saturation=0.2, pressure=100e5, temperature=350.0, salinity=1.0
    ) -> None:
        self.sw = water_saturation
        self.p = pressure
        self.T = temperature
        self.salinity = salinity


class NumericalSettings:
    """
    Numerical settings for the simulation.
    This includes the accuracy parameters for the solver
    """

    def __init__(
        self,
        dp_allowed=100,
        dsw_allowed=0.05,
        eps_p=1e-5,
        eps_sw=1e-5,
        simulation_time=18000.0,
        time_step=100.0,
        time_step_multiplier=10.0,
    ) -> None:
        self.dp_allowed = dp_allowed
        self.dsw_allowed = dsw_allowed
        self.eps_p = eps_p
        self.eps_sw = eps_sw
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.time_step_multiplier = time_step_multiplier


class FloodingConditions:
    """
    Operational conditions for the reservoir. This includes the injection rate and
    injection/production pressures.

    Args:
        injection_velocity (float, optional): Injection velocity in m/s. Default is 1e-5.
        injection_pressure (float, optional): Injection pressure in Pa. Default is 100e5.
        production_pressure (float, optional): Production pressure in Pa. Default is 50e5.
    """

    def __init__(
        self,
        injection_rate_ml_min: float = 0.1,
        injection_pressure: float = 100e5,
        production_pressure: float = 50e5,
        injection_sw=1.0,
        active_rate=True,
    ):
        self.injection_rate_ml_min = injection_rate_ml_min
        self.injection_pressure = injection_pressure
        self.production_pressure = production_pressure
        self.active_rate = active_rate
        self.injection_sw = injection_sw


class ImbibitionConditions:
    def __init__(self, top_saturations, bottom_saturations, side_saturation):
        self.top_saturations = top_saturations


class CoreModel1D:
    def __init__(
        self,
        reservoir: Reservoir,
        operational_conditions: FloodingConditions,
        core: CorePlug,
        Nx: int = 50,
        dp_allowed=100,
        dsw_allowed=0.05,
        eps_p=1e-5,
        eps_sw=1e-5,
    ):
        m = createMesh1D(Nx, core.core_lengthlength)
        self.pore_volume = core.pore_volume
        self.injection_velocity = operational_conditions.injection_velocity
        self.dp_allowed = dp_allowed
        self.dsw_allowed = dsw_allowed
        self.eps_p = eps_p
        self.eps_sw = eps_sw
        self.domain = m
        self.perm_field = createCellVariable(m, reservoir.permeability)
        self.poros_field = createCellVariable(m, reservoir.porosity)
        mu_water = createCellVariable(m, reservoir.fluids.water_viscosity)
        mu_oil = createCellVariable(m, reservoir.fluids.oil_viscosity)
        self.water_mobility_max = geometricMean(self.perm_field / mu_water)
        self.oil_mobility_max = geometricMean(self.perm_field / mu_oil)
        self.rel_perm = reservoir.rel_perm

        BCp = createBC(m)  # Neumann BC for pressure
        # back pressure boundary condition
        BCp.right.a[:] = 0.0
        BCp.right.b[:] = 1.0
        BCp.right.c[:] = operational_conditions.production_pressure
        # injection boundary
        BCp.left.a[:] = self.water_mobility_max.xvalue[0]
        BCp.left.b[:] = 0.0
        BCp.left.c[:] = -operational_conditions.injection_velocity
        # saturation left boundary
        BCs = createBC(m)  # Neumann BC for saturation
        BCs.left.a[:] = 0.0
        BCs.left.b[:] = 1.0
        BCs.left.c[:] = operational_conditions.injection_sw

        self.pressure_bc = BCp
        self.saturation_bc = BCs
        self.initial_pressure = createCellVariable(m, reservoir.initial_p, BCp)
        self.initial_sw = createCellVariable(m, reservoir.initial_sw, BCs)
        self.pressure = self.initial_pressure
        self.saturation = self.initial_sw

    def calculate_water_mobility(self, sw_face):
        return self.water_mobility_max * faceeval(self.rel_perm.krw, sw_face)

    def calculate_oil_mobility(self, sw_face):
        return self.oil_mobility_max * faceeval(self.rel_perm.kro, sw_face)

    def calculate_water_mobility_derivative(self, sw_face):
        return self.water_mobility_max * faceeval(self.rel_perm.dkrwdsw, sw_face)

    def calculate_oil_mobility_derivative(self, sw_face):
        return self.oil_mobility_max * faceeval(self.rel_perm.dkrodsw, sw_face)

    def calculate_water_velocity(self, sw_face):
        return self.calculate_water_mobility(sw_face) * gradientTerm(self.pressure)

    def simulate(self, final_time=100000.0, dt=1000.0):
        t = 0.0
        Nxyz = np.prod(self.domain.dims)
        while t < final_time:
            error_p = 1e5
            error_sw = 1e5
            loop_count = 0
            while error_p > self.eps_p or error_sw > self.eps_sw:
                loop_count += 1
                if loop_count > 10:
                    break
                pgrad = gradientTerm(self.pressure)
                sw_face = upwindMean(self.saturation, -pgrad)
                labdao = self.calculate_oil_mobility(sw_face)
                labdaw = self.calculate_water_mobility(sw_face)
                dlabdaodsw = self.calculate_oil_mobility_derivative(sw_face)
                dlabdawdsw = self.calculate_water_mobility_derivative(sw_face)
                labda = labdao + labdaw
                dlabdadsw = dlabdaodsw + dlabdawdsw
                Mdiffp1 = diffusionTerm(-labda)
                Mdiffp2 = diffusionTerm(-labdaw)
                Mconvsw1 = convectionUpwindTerm(-dlabdadsw * pgrad)
                Mconvsw2 = convectionUpwindTerm(-dlabdawdsw * pgrad)
                [Mtranssw2, RHStrans2] = transientTerm(
                    self.initial_sw, dt, self.poros_field
                )
                RHS1 = divergenceTerm(-dlabdadsw * sw_face * pgrad)
                RHS2 = divergenceTerm(-dlabdawdsw * sw_face * pgrad)
                [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
                [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
                M = vstack(
                    [
                        hstack([Mdiffp1 + Mbcp, Mconvsw1]),
                        hstack([Mdiffp2, Mconvsw2 + Mtranssw2 + Mbcsw]),
                    ]
                )
                RHS = np.hstack([RHS1 + RHSbcp, RHS2 + RHStrans2 + RHSbcsw])
                x = spsolve(M, RHS)
                p_new = np.reshape(x[0 : (Nxyz + 2)], (Nxyz + 2))
                sw_new = np.reshape(x[(Nxyz + 2) :], (Nxyz + 2))
                error_p = np.max(np.abs((p_new - self.pressure.value[:]) / p_new))
                error_sw = np.max(np.abs(sw_new - self.saturation.value[:]))
                self.pressure.value[:] = p_new
                self.saturation.value[:] = sw_new
            if loop_count > 10:
                self.pressure.value[:] = self.initial_pressure.value[:]
                self.saturation.value[:] = self.initial_sw.value[:]
                dt = dt / 5
                continue

            dsw = np.max(abs(sw_new[:] - self.initial_sw.value[:]) / sw_new[:])
            t += dt
            dt = np.min([dt * (self.dsw_allowed / dsw), 2 * dt, final_time - t])
            self.initial_pressure.value[:] = self.pressure.value[:]
            self.initial_sw.value[:] = self.saturation.value[:]


class CoreFlooding1D:
    def __init__(
        self,
        rel_perm: RelativePermeability,
        pc: CapillaryPressure,
        core_plug: CorePlug,
        fluids: Fluids,
        IC: InitialConditions,
        BC: FloodingConditions,
        numerical_params: NumericalSettings,
        Nx: int = 30,
    ) -> None:
        self.rel_perm = rel_perm
        self.core_plug = core_plug
        self.pc = pc
        self.fluids = fluids
        self.IC = IC
        self.BC = BC
        self.numerical_params = numerical_params
        m = createMesh1D(Nx, core_plug.core_length)
        self.domain = m
        k = createCellVariable(m, core_plug.permeability)
        self.permeability = k
        phi = createCellVariable(m, core_plug.porosity)
        self.porosity = phi
        self.water_viscosity = createCellVariable(m, fluids.water_viscosity)
        self.oil_viscosity = createCellVariable(m, fluids.oil_viscosity)
        p0 = IC.p  # [Pa] pressure
        p_back = BC.production_pressure  # [Pa] pressure
        u_inj = BC.injection_rate_ml_min / (
            core_plug.cross_sectional_area * 1e4
        )  # [cm/min]
        u_inj = u_inj / (60 * 100)  # [m/s]
        self.u_inj = u_inj
        sw0 = IC.sw  # initial saturation
        BCp = createBC(m)  # Neumann BC for pressure
        BCs = createBC(m)  # Neumann BC for saturation
        BCp.right.a[:] = 0.0
        BCp.right.b[:] = 1.0
        BCp.right.c[:] = p_back
        BCp.left.a[:] = 1.0
        BCp.left.b[:] = 0.0
        BCp.left.c[:] = (
            -u_inj * self.water_viscosity.value[0] / self.permeability.value[0]
        )
        BCs.left.a[:] = 0.0
        BCs.left.b[:] = 1.0
        BCs.left.c[:] = 1.0
        self.initial_pressure = createCellVariable(m, p0, BCp)
        self.initial_sw = createCellVariable(m, sw0, BCs)
        self.final_pressure = createCellVariable(m, p0, BCp)
        self.final_sw = createCellVariable(m, sw0, BCs)
        self.pressure_bc = BCp
        self.saturation_bc = BCs

    def simulate_no_pc(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        pgrad = gradientTerm(p)
        sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
        labdao = lo * faceeval(self.rel_perm.kro, sw_face)
        labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
        labda = labdao + labdaw
        Mdiffp1 = diffusionTerm(-labda)
        RHS1 = RHSbcp  # with capillary
        p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
        p.update_value(p_new)
        dp_hist = np.array([p_new.value[0:2].mean() - p_new.value[-2:].mean()])
        t = 0.0
        while t < t_end:
            error_sw = 1e5
            while 1:  # loop condition is checked inside
                pgrad = gradientTerm(p)
                sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                labda = labdao + labdaw
                Mdiffp1 = diffusionTerm(-labda)
                RHS1 = RHSbcp  # with capillary
                p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                # solve for Sw
                pgrad = gradientTerm(p_new)
                uw = -labdaw * pgrad
                RHS_sw = -divergenceTerm(uw)
                sw_new = solveExplicitPDE(
                    sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                )
                error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                # calculate new time step
                # assign new values of p and sw
                if error_sw > dsw_alwd:
                    dt = dt * (dsw_alwd / error_sw)
                else:
                    t = t + dt
                    p.update_value(p_new)
                    sw.update_value(sw_new)
                    p_old.update_value(p)
                    sw_old.update_value(sw)
                    dt = min(
                        dt * (dsw_alwd / error_sw),
                        self.numerical_params.time_step_multiplier * dt,
                    )
                    break
            # calculate recovery factor
            rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
            t_hist = np.append(t_hist, t)
            dp_hist = np.append(
                dp_hist, p_new.value[0:2].mean() - p_new.value[-2:].mean()
            )
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact, dp_hist

    def simulate_with_pc(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        # sol = root(self.pc.pc_imb, self.pc.sw_pc0) # saturation at which pc=0
        # sw_pc0 = sol.x # saturation at which pc=0
        sw_pc0 = self.pc.sw_pc0
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        dp_hist = np.array([0])
        t = 0.0
        while t < t_end:
            error_sw = 1e5
            while 1:  # loop condition is checked inside
                pgrad = gradientTerm(p)
                sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                sw_ave=arithmeticMean(sw)
                sw_temp = sw.copy()
                sw_temp.bc_to_ghost()
                pc_cell = funceval(self.pc.pc_imb, sw_temp)
                pcgrad = gradientTermFixedBC(pc_cell)
                labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                labda = labdao + labdaw
                Mdiffp1 = diffusionTerm(-labda)
                RHSpc1=divergenceTerm(labdao*pcgrad)
                RHS1 = RHSbcp+RHSpc1  # with capillary
                p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                # solve for Sw
                pgrad = gradientTerm(p_new)
                uw = -labdaw * pgrad
                RHS_sw = -divergenceTerm(uw)
                sw_new = solveExplicitPDE(
                    sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                )
                error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                # calculate new time step
                # assign new values of p and sw
                if error_sw > dsw_alwd:
                    dt = dt * (dsw_alwd / error_sw)
                else:
                    t = t + dt
                    p.update_value(p_new)
                    sw.update_value(sw_new)
                    p_old.update_value(p)
                    sw_old.update_value(sw)
                    dt = min(
                        dt * (dsw_alwd / error_sw),
                        self.numerical_params.time_step_multiplier * dt,
                    )
                    break
            if sw_ave.xvalue[-1]>=sw_pc0:
                self.saturation_bc.right.a[:] = 0. 
                self.saturation_bc.right.b[:] = 1. 
                self.saturation_bc.right.c[:]=sw_pc0
            # calculate recovery factor
            rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
            t_hist = np.append(t_hist, t)
            dp_hist = np.append(
                dp_hist, p_new.value[0:2].mean() - p_new.value[-2:].mean()
            )
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact, dp_hist

class CoreFlooding2D:
    def __init__(
        self,
        rel_perm: RelativePermeability,
        pc: CapillaryPressure,
        core_plug: CorePlug,
        fluids: Fluids,
        IC: InitialConditions,
        BC: FloodingConditions,
        numerical_params: NumericalSettings,
        Nx: int = 30, Nr = 10
    ) -> None:
        """
        define a 2D core flooding model
        TODO:
        1. add heterogeneous permeability and porosity
        2. add heterogeneous capillary pressure
        3. add heterogeneous initial conditions
        4. add heterogeneous boundary conditions
        """
        self.rel_perm = rel_perm
        self.core_plug = core_plug
        self.pc = pc
        self.fluids = fluids
        self.IC = IC
        self.BC = BC
        self.numerical_params = numerical_params
        m = createMeshCylindrical2D(Nr, Nx, core_plug.diameter/2, core_plug.core_length)
        self.domain = m
        k = createCellVariable(m, core_plug.permeability)
        self.permeability = k
        phi = createCellVariable(m, core_plug.porosity)
        self.porosity = phi
        self.water_viscosity = createCellVariable(m, fluids.water_viscosity)
        self.oil_viscosity = createCellVariable(m, fluids.oil_viscosity)
        lw = geometricMean(self.permeability / self.water_viscosity)
        p0 = IC.p  # [Pa] pressure
        p_back = BC.production_pressure  # [Pa] pressure
        u_inj = BC.injection_rate_ml_min / (
            core_plug.cross_sectional_area * 1e4
        )  # [cm/min]
        u_inj = u_inj / (60 * 100)  # [m/s]
        self.u_inj = u_inj
        sw0 = IC.sw  # initial saturation
        BCp = createBC(m)  # Neumann BC for pressure
        BCs = createBC(m)  # Neumann BC for saturation
        BCp.top.a[:] = 0.0
        BCp.top.b[:] = 1.0
        BCp.top.c[:] = p_back
        BCp.bottom.a[:] = lw.yvalue[:,0]
        BCp.bottom.b[:] = 0.0
        BCp.bottom.c[:] = (
            -u_inj
        )
        BCs.bottom.a[:] = 0.0
        BCs.bottom.b[:] = 1.0
        BCs.bottom.c[:] = 1.0
        self.initial_pressure = createCellVariable(m, p0, BCp)
        self.initial_sw = createCellVariable(m, sw0, BCs)
        self.final_pressure = createCellVariable(m, p0, BCp)
        self.final_sw = createCellVariable(m, sw0, BCs) 
        self.pressure_bc = BCp
        self.saturation_bc = BCs

    def simulate_no_pc(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        pgrad = gradientTerm(p)
        sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
        labdao = lo * faceeval(self.rel_perm.kro, sw_face)
        labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
        labda = labdao + labdaw
        Mdiffp1 = diffusionTerm(-labda)
        RHS1 = RHSbcp  # with capillary
        p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
        p_ave = arithmeticMean(p_new)
        dp_hist = np.array([p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean()])
        # p.update_value(p_new)
        t = 0.0
        while t < t_end:
            error_sw = 1e5
            while 1:  # loop condition is checked inside
                pgrad = gradientTerm(p)
                sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                labda = labdao + labdaw
                Mdiffp1 = diffusionTerm(-labda)
                RHS1 = RHSbcp  # with capillary
                p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                # solve for Sw
                pgrad = gradientTerm(p_new)
                uw = -labdaw * pgrad
                RHS_sw = -divergenceTerm(uw)
                sw_new = solveExplicitPDE(
                    sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                )
                error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                # calculate new time step
                # assign new values of p and sw
                if error_sw > dsw_alwd:
                    dt = dt * (dsw_alwd / error_sw)
                else:
                    t = t + dt
                    p.update_value(p_new)
                    sw.update_value(sw_new)
                    p_old.update_value(p)
                    sw_old.update_value(sw)
                    dt = min(
                        dt * (dsw_alwd / error_sw),
                        self.numerical_params.time_step_multiplier * dt,
                    )
                    break
            # calculate recovery factor
            rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
            t_hist = np.append(t_hist, t)
            p_ave = arithmeticMean(p_new)
            dp_hist = np.append(
                dp_hist, p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean() 
            )
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact, dp_hist

    def simulate_with_pc(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        # sol = root(self.pc.pc_imb, self.pc.sw_pc0) # saturation at which pc=0
        # sw_pc0 = sol.x # saturation at which pc=0
        sw_pc0 = self.pc.sw_pc0
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        pgrad = gradientTerm(p)
        sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
        labdao = lo * faceeval(self.rel_perm.kro, sw_face)
        labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
        labda = labdao + labdaw
        Mdiffp1 = diffusionTerm(-labda)
        RHS1 = RHSbcp  # with capillary
        p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
        p_ave = arithmeticMean(p_new)
        dp_hist = np.array([p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean()])
        # p.update_value(p_new)
        t = 0.0
        while t < t_end:
            error_sw = 1e5
            while 1:  # loop condition is checked inside
                pgrad = gradientTerm(p)
                sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                sw_ave=arithmeticMean(sw)
                sw_temp = sw.copy()
                BC2GhostCells(sw_temp)
                pc_cell = funceval(self.pc.pc_imb, sw_temp)
                pcgrad = gradientTermFixedBC(pc_cell)
                labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                labda = labdao + labdaw
                Mdiffp1 = diffusionTerm(-labda)
                RHSpc1=divergenceTerm(labdao*pcgrad)
                RHS1 = RHSbcp+RHSpc1  # with capillary
                p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                # solve for Sw
                pgrad = gradientTerm(p_new)
                uw = -labdaw * pgrad
                RHS_sw = -divergenceTerm(uw)
                sw_new = solveExplicitPDE(
                    sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                )
                error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                # calculate new time step
                # assign new values of p and sw
                if error_sw > dsw_alwd:
                    dt = dt * (dsw_alwd / error_sw)
                else:
                    t = t + dt
                    p.update_value(p_new)
                    sw.update_value(sw_new)
                    p_old.update_value(p)
                    sw_old.update_value(sw)
                    dt = min(
                        dt * (dsw_alwd / error_sw),
                        self.numerical_params.time_step_multiplier * dt,
                    )
                    break
            if sw_ave.yvalue[:,-1].mean()>=sw_pc0:
                self.saturation_bc.top.a[:] = 0. 
                self.saturation_bc.top.b[:] = 1. 
                self.saturation_bc.top.c[:]=sw_pc0
            # calculate recovery factor
            rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
            t_hist = np.append(t_hist, t)
            p_ave = arithmeticMean(p_new)
            dp_hist = np.append(
                dp_hist, p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean() 
            )
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact, dp_hist
    
    

class CoreImbibition:
    def __init__(
        self,
        rel_perm: RelativePermeability,
        pc: CapillaryPressure,
        pc_new: CapillaryPressure,
        core_plug: CorePlug,
        fluids: Fluids,
        IC: InitialConditions,
        BC: FloodingConditions,
        numerical_params: NumericalSettings,
        Nx: int = 30, Nr = 10,
        Nx_film = 20,
        c0 = 0.0, c_in = 1.0,
        L_film = 10e-6, D_film=1e-12,
        D_bulk = 1e-9
    ) -> None:
        """
        define a 2D spontaneous imbibition model
        TODO:
        1. add heterogeneous permeability and porosity
        2. add heterogeneous capillary pressure
        3. add heterogeneous initial conditions
        4. add heterogeneous boundary conditions
        """
        self.rel_perm = rel_perm
        self.core_plug = core_plug
        self.pc = pc
        self.pc_new = pc_new
        self.fluids = fluids
        self.IC = IC
        self.BC = BC
        self.numerical_params = numerical_params
        m = createMeshCylindrical2D(Nr, Nx, core_plug.diameter/2, core_plug.core_length)
        m_film = createMesh1D(Nx_film, L_film)
        self.domain = m
        self.domain_film = m_film
        k = createCellVariable(m, core_plug.permeability)
        self.permeability = k
        phi = createCellVariable(m, core_plug.porosity)
        self.D_cell = createCellVariable(m, D_bulk)
        self.D_film = createCellVariable(m_film, D_film)
        self.porosity = phi
        self.water_viscosity = createCellVariable(m, fluids.water_viscosity)
        self.oil_viscosity = createCellVariable(m, fluids.oil_viscosity)
        lw = geometricMean(self.permeability / self.water_viscosity)
        p0 = IC.p  # [Pa] pressure
        p_back = BC.production_pressure  # [Pa] pressure
        sw0 = IC.sw  # initial saturation
        BCp = createBC(m)  # Neumann BC for pressure
        BCs = createBC(m)  # Neumann BC for saturation
        BCc = createBC(m) # Neumann BC for concentration
        BCf = createBC(m_film) # Neumann BC for concentration
        BCf.right.a[:] = 0.0
        BCf.right.b[:] = 1.0
        BCf.right.c[:] = c0
        
        BCc.top.a[:] = 0.0
        BCc.top.b[:] = 1.0
        BCc.top.c[:] = c_in
        BCc.bottom.a[:] = 0.0
        BCc.bottom.b[:] = 1.0
        BCc.bottom.c[:] = c_in
        BCc.right.a[:] = 0.0
        BCc.right.b[:] = 1.0
        BCc.right.c[:] = c_in

        BCp.top.a[:] = 0.0
        BCp.top.b[:] = 1.0
        BCp.top.c[:] = p_back
        BCp.bottom.a[:] = 0.0
        BCp.bottom.b[:] = 1.0
        BCp.bottom.c[:] = p_back
        BCp.right.a[:] = 0.0
        BCp.right.b[:] = 1.0
        BCp.right.c[:] = p_back
        if hasattr(self.pc, 'sw_pc0'):
            sw_pc0 = self.pc.sw_pc0
        else:
            sw_pc0 = root(self.pc.pc_imb, sw0).x

        BCs.bottom.a[:] = 0.0
        BCs.bottom.b[:] = 1.0
        BCs.bottom.c[:] = sw_pc0
        BCs.top.a[:] = 0.0
        BCs.top.b[:] = 1.0
        BCs.top.c[:] = sw_pc0
        BCs.right.a[:] = 0.0
        BCs.right.b[:] = 1.0
        BCs.right.c[:] = sw_pc0

        self.initial_pressure = createCellVariable(m, p0, BCp)
        self.initial_sw = createCellVariable(m, sw0, BCs)
        self.final_pressure = createCellVariable(m, p0, BCp)
        self.final_sw = createCellVariable(m, sw0, BCs) 
        self.pressure_bc = BCp
        self.saturation_bc = BCs
        self.concentration_bc = BCc
        self.film_bc = BCf
        self.initial_c = createCellVariable(m, c0, BCc)
        self.c_film_cells = np.array([[createCellVariable(m_film, c0, BCf) for j in range(Nx)] for i in range(Nr)])
        self.wet_change = createCellVariable(m, 0.0)
    def simulate(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        pgrad = gradientTerm(p)
        sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
        labdao = lo * faceeval(self.rel_perm.kro, sw_face)
        labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
        labda = labdao + labdaw
        Mdiffp1 = diffusionTerm(-labda)
        RHS1 = RHSbcp  # with capillary
        p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
        # diffusion term for concentration
        Mdiffc = diffusionTerm(geometricMean(self.D_cell))
        Mdiffc_film = diffusionTerm(geometricMean(self.D_film))
        Mbcc, RHSbcc = boundaryConditionTerm(self.concentration_bc)
        BC_temp = createBC(self.domain)
        # p_ave = arithmeticMean(p_new)
        # dp_hist = np.array([p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean()])
        # p.update_value(p_new)
        t = 0.0
        with tqdm(total=t_end) as pbar:
            while t < t_end:
                pbar.update(dt)
                error_sw = 1e5
                while 1:  # loop condition is checked inside
                    pgrad = gradientTerm(p)
                    sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                    # sw_ave=arithmeticMean(sw)
                    sw_temp = sw.copy()
                    BC2GhostCells(sw_temp)
                    pc_cell_old = funceval(self.pc.pc_imb, sw_temp)
                    pc_cell_new = funceval(self.pc_new.pc_imb, sw_temp)
                    pc_cell = self.wet_change*pc_cell_new+(1-self.wet_change)*pc_cell_old
                    pcgrad = gradientTermFixedBC(pc_cell)
                    labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                    labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                    labda = labdao + labdaw
                    Mdiffp1 = diffusionTerm(-labda)
                    RHSpc1=divergenceTerm(labdao*pcgrad)
                    RHS1 = RHSbcp+RHSpc1  # with capillary
                    p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                    # solve for Sw
                    pgrad = gradientTerm(p_new)
                    uw = -labdaw * pgrad
                    RHS_sw = -divergenceTerm(uw)
                    sw_new = solveExplicitPDE(
                        sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                    )
                    error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                    # calculate new time step
                    # assign new values of p and sw
                    if error_sw > dsw_alwd:
                        dt = dt * (dsw_alwd / error_sw)
                    else:
                        # calculate concentration profile
                        Mtc, RHStc = transientTerm(self.initial_c, dt, self.porosity*sw_new)
                        Mlinc = linearSourceTerm(self.porosity*(sw_new-sw_old)/dt)
                        Mconv = convectionUpwindTerm(uw)
                        c_new = solvePDE(self.domain, Mtc+Mlinc+Mconv-Mdiffc+Mbcc, RHStc+RHSbcc)
                        # calculate film diffusion
                        nx, ny = self.domain.dims
                        for i in range(nx):
                            for j in range(ny):
                                Mtcf, RHStcf = transientTerm(self.c_film_cells[i,j], dt, 1.0)
                                self.film_bc.right.c[:] = c_new.value[i+1,j+1]
                                Mbcf, RHSbcf = boundaryConditionTerm(self.film_bc)
                                c_temp = solvePDE(self.domain_film, Mtcf-Mdiffc_film+Mbcf, RHStcf+RHSbcf)
                                self.c_film_cells[i,j].update_value(c_temp)
                                # use film concentration to calculate wettability change
                                if self.c_film_cells[i,j].value[2]>0.5:
                                    self.wet_change.value[i+1,j+1] = 1.0
                        self.wet_change.update_bc_cells(BC_temp)

                        t = t + dt
                        p.update_value(p_new)
                        self.initial_c.update_value(c_new)
                        sw.update_value(sw_new)
                        p_old.update_value(p)
                        sw_old.update_value(sw)
                        dt = min(
                            dt * (dsw_alwd / error_sw),
                            self.numerical_params.time_step_multiplier * dt,
                        )
                        break
                # calculate recovery factor
                rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
                t_hist = np.append(t_hist, t)
                # p_ave = arithmeticMean(p_new)
                # dp_hist = np.append(
                #     dp_hist, p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean() 
                # )
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact
    
    def simulate_no_film(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        pgrad = gradientTerm(p)
        sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
        labdao = lo * faceeval(self.rel_perm.kro, sw_face)
        labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
        labda = labdao + labdaw
        Mdiffp1 = diffusionTerm(-labda)
        RHS1 = RHSbcp  # with capillary
        p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
        # diffusion term for concentration
        Mdiffc = diffusionTerm(geometricMean(self.D_cell))
        Mdiffc_film = diffusionTerm(geometricMean(self.D_film))
        Mbcc, RHSbcc = boundaryConditionTerm(self.concentration_bc)
        BC_temp = createBC(self.domain)
        # p_ave = arithmeticMean(p_new)
        # dp_hist = np.array([p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean()])
        # p.update_value(p_new)
        t = 0.0
        with tqdm(total=t_end) as pbar:
            while t < t_end:
                pbar.update(dt)
                error_sw = 1e5
                while 1:  # loop condition is checked inside
                    pgrad = gradientTerm(p)
                    sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                    # sw_ave=arithmeticMean(sw)
                    sw_temp = sw.copy()
                    BC2GhostCells(sw_temp)
                    # pc_cell_old = funceval(self.pc.pc_imb, sw_temp)
                    pc_cell = funceval(self.pc_new.pc_imb, sw_temp)
                    # pc_cell = self.wet_change*pc_cell_new+(1-self.wet_change)*pc_cell_old
                    pcgrad = gradientTermFixedBC(pc_cell)
                    labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                    labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                    labda = labdao + labdaw
                    Mdiffp1 = diffusionTerm(-labda)
                    RHSpc1=divergenceTerm(labdao*pcgrad)
                    RHS1 = RHSbcp+RHSpc1  # with capillary
                    p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                    # solve for Sw
                    pgrad = gradientTerm(p_new)
                    uw = -labdaw * pgrad
                    RHS_sw = -divergenceTerm(uw)
                    sw_new = solveExplicitPDE(
                        sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                    )
                    error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                    # calculate new time step
                    # assign new values of p and sw
                    if error_sw > dsw_alwd:
                        dt = dt * (dsw_alwd / error_sw)
                    else:
                        # calculate concentration profile
                        # Mtc, RHStc = transientTerm(self.initial_c, dt, self.porosity*sw_new)
                        # Mlinc = linearSourceTerm(self.porosity*(sw_new-sw_old)/dt)
                        # Mconv = convectionUpwindTerm(uw)
                        # c_new = solvePDE(self.domain, Mtc+Mlinc+Mconv-Mdiffc+Mbcc, RHStc+RHSbcc)
                        # # calculate film diffusion
                        # nx, ny = self.domain.dims
                        # for i in range(nx):
                        #     for j in range(ny):
                        #         Mtcf, RHStcf = transientTerm(self.c_film_cells[i,j], dt, 1.0)
                        #         self.film_bc.right.c[:] = c_new.value[i+1,j+1]
                        #         Mbcf, RHSbcf = boundaryConditionTerm(self.film_bc)
                        #         self.c_film_cells[i,j] = solvePDE(self.domain_film, Mtcf-Mdiffc_film+Mbcf, RHStcf+RHSbcf)
                        #         # use film concentration to calculate wettability change
                        #         if self.c_film_cells[i,j].value[-2]>0.5:
                        #             self.wet_change.value[i+1,j+1] = 1.0
                        # self.wet_change.update_bc_cells(BC_temp)

                        t = t + dt
                        p.update_value(p_new)
                        # self.initial_c.update_value(c_new)
                        sw.update_value(sw_new)
                        p_old.update_value(p)
                        sw_old.update_value(sw)
                        dt = min(
                            dt * (dsw_alwd / error_sw),
                            self.numerical_params.time_step_multiplier * dt,
                        )
                        break
                # calculate recovery factor
                rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
                t_hist = np.append(t_hist, t)
                # p_ave = arithmeticMean(p_new)
                # dp_hist = np.append(
                #     dp_hist, p_ave.yvalue[:, 0].mean() - p_ave.yvalue[:,-1].mean() 
                # )
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact

class CoreModel2D:
    def __init__(self) -> None:
        pass


class CoreModel3D:
    def __init__(self) -> None:
        pass

class Imbibition1D:
    def __init__(
        self,
        rel_perm: RelativePermeability,
        pc: CapillaryPressure,
        core_plug: CorePlug,
        fluids: Fluids,
        IC: InitialConditions,
        BC: FloodingConditions,
        numerical_params: NumericalSettings,
        Nx: int = 30,
        Nx_free = 10,
        perm_free = 100e-12,
        poros_free = 1.0
    ) -> None:
        self.Nx_free = 10
        self.Nx = Nx
        self.rel_perm = rel_perm
        self.rel_perm2 = RelativePermeability(swc=eps(), sor=eps(), kro0=1.0, no=1.0, krw0=1.0, nw=1.0)
        self.core_plug = core_plug
        self.pc = pc
        self.fluids = fluids
        self.IC = IC
        self.BC = BC
        self.numerical_params = numerical_params
        m = createMesh1D(Nx+Nx_free, core_plug.core_length)
        self.domain = m
        k_val = np.hstack([core_plug.permeability*np.ones(Nx), perm_free*np.ones(Nx_free)])
        k = createCellVariable(m, k_val)
        self.permeability = k
        phi = createCellVariable(m, 
                                 np.hstack([core_plug.porosity*np.ones(Nx),
                                     poros_free*np.ones(Nx_free)]))
        self.porosity = phi
        self.water_viscosity = createCellVariable(m, fluids.water_viscosity)
        self.oil_viscosity = createCellVariable(m, fluids.oil_viscosity)
        p0 = IC.p  # [Pa] pressure
        p_back = BC.production_pressure  # [Pa] pressure
        sw0 = IC.sw  # initial saturation
        sw_val = np.hstack([sw0*np.ones(Nx), np.ones(Nx_free)])
        BCp = createBC(m)  # Neumann BC for pressure
        BCs = createBC(m)  # Neumann BC for saturation
        BCp.right.a[:] = 0.0
        BCp.right.b[:] = 1.0
        BCp.right.c[:] = p_back
        self.initial_pressure = createCellVariable(m, p0, BCp)
        self.initial_sw = createCellVariable(m, sw_val, BCs)
        self.final_pressure = createCellVariable(m, p0, BCp)
        self.final_sw = createCellVariable(m, sw_val, BCs)
        self.pressure_bc = BCp
        self.saturation_bc = BCs

    def simulate_no_pc(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        pgrad = gradientTerm(p)
        sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
        labdao = lo * faceeval(self.rel_perm.kro, sw_face)
        labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
        labda = labdao + labdaw
        Mdiffp1 = diffusionTerm(-labda)
        RHS1 = RHSbcp  # with capillary
        p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
        p.update_value(p_new)
        dp_hist = np.array([p_new.value[0:2].mean() - p_new.value[-2:].mean()])
        t = 0.0
        while t < t_end:
            error_sw = 1e5
            while 1:  # loop condition is checked inside
                pgrad = gradientTerm(p)
                sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                labda = labdao + labdaw
                Mdiffp1 = diffusionTerm(-labda)
                RHS1 = RHSbcp  # with capillary
                p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                # solve for Sw
                pgrad = gradientTerm(p_new)
                uw = -labdaw * pgrad
                RHS_sw = -divergenceTerm(uw)
                sw_new = solveExplicitPDE(
                    sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                )
                error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                # calculate new time step
                # assign new values of p and sw
                if error_sw > dsw_alwd:
                    dt = dt * (dsw_alwd / error_sw)
                else:
                    t = t + dt
                    p.update_value(p_new)
                    sw.update_value(sw_new)
                    p_old.update_value(p)
                    sw_old.update_value(sw)
                    dt = min(
                        dt * (dsw_alwd / error_sw),
                        self.numerical_params.time_step_multiplier * dt,
                    )
                    break
            # calculate recovery factor
            rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
            t_hist = np.append(t_hist, t)
            dp_hist = np.append(
                dp_hist, p_new.value[0:2].mean() - p_new.value[-2:].mean()
            )
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact, dp_hist

    def simulate_with_pc(self):
        t_end = self.numerical_params.simulation_time  # [s] final simulation time
        # eps_p = self.numerical_params.eps_p # pressure accuracy
        # eps_sw = self.numerical_params.eps_sw # saturation accuracy
        # sol = root(self.pc.pc_imb, self.pc.sw_pc0) # saturation at which pc=0
        # sw_pc0 = sol.x # saturation at which pc=0
        sw_pc0 = self.pc.sw_pc0
        dsw_alwd = self.numerical_params.dsw_allowed
        # dp_alwd= self.numerical_params.dp_allowed # Pa
        dt = self.numerical_params.time_step  # [s] initial time step
        [Mbcp, RHSbcp] = boundaryConditionTerm(self.pressure_bc)
        # [Mbcsw, RHSbcsw] = boundaryConditionTerm(self.saturation_bc)
        lw = geometricMean(self.permeability / self.water_viscosity)
        lo = geometricMean(self.permeability / self.oil_viscosity)
        # initial conditions; will clean up by creating a copy of the initial pressure and saturation
        sw_old = createCellVariable(self.domain, 0.0)
        sw_old.update_value(self.initial_sw)
        p_old = createCellVariable(self.domain, 0.0)
        p_old.update_value(self.initial_pressure)
        p = createCellVariable(self.domain, 0.0)
        p.update_value(self.initial_pressure)
        sw = createCellVariable(self.domain, 0.0)
        sw.update_value(self.initial_sw)
        uw = -gradientTerm(p_old)  # an estimation of the water velocity
        oil_init = domainInt(1 - sw_old)
        rec_fact = np.array([0])
        t_hist = np.array([0])
        dp_hist = np.array([0])
        sw_border = np.array(sw.value[self.Nx])
        t = 0.0
        with tqdm(total=t_end) as pbar:
            while t < t_end:
                error_sw = 1e5
                # print(t/t_end*100)
                while 1:  # loop condition is checked inside
                    pgrad = gradientTerm(p)
                    sw_face = upwindMean(sw, -pgrad)  # average value of water saturation
                    sw_ave=arithmeticMean(sw)
                    sw_temp = sw.copy()
                    sw_temp.bc_to_ghost()
                    pc_cell = funceval(self.pc.pc_imb, sw_temp)
                    pc_cell.value[self.Nx+1:] = 0.0
                    pcgrad = gradientTermFixedBC(pc_cell)
                    labdao = lo * faceeval(self.rel_perm.kro, sw_face)
                    labdao2 = lo * faceeval(self.rel_perm2.kro, sw_face)
                    labdao.xvalue[self.Nx+1:] = labdao2.xvalue[self.Nx+1:]
                    labdaw = lw * faceeval(self.rel_perm.krw, sw_face)
                    labdaw2 = lw * faceeval(self.rel_perm2.krw, sw_face)
                    labdaw.xvalue[self.Nx+1:] = labdaw2.xvalue[self.Nx+1:]
                    labda = labdao + labdaw
                    Mdiffp1 = diffusionTerm(-labda)
                    RHSpc1=divergenceTerm(labdao*pcgrad)
                    RHS1 = RHSbcp+RHSpc1  # with capillary
                    p_new = solvePDE(self.domain, Mdiffp1 + Mbcp, RHS1)
                    # solve for Sw
                    pgrad = gradientTerm(p_new)
                    uw = -labdaw * pgrad
                    RHS_sw = -divergenceTerm(uw)
                    sw_new = solveExplicitPDE(
                        sw_old, dt, RHS_sw / self.porosity.value.ravel(), self.saturation_bc
                    )
                    error_sw = np.max(np.abs(sw_new.internalCells() - sw.internalCells()))
                    # calculate new time step
                    # assign new values of p and sw
                    if error_sw > dsw_alwd:
                        dt = dt * (dsw_alwd / error_sw)
                    else:
                        t = t + dt
                        p.update_value(p_new)
                        sw.update_value(sw_new)
                        p_old.update_value(p)
                        sw_old.update_value(sw)
                        dt = min(
                            dt * (dsw_alwd / error_sw),
                            self.numerical_params.time_step_multiplier * dt,
                        )
                        break
                    pbar.update(dt)
                    # calculate recovery factor
                    rec_fact = np.append(rec_fact, (oil_init - domainInt(1 - sw)) / oil_init)
                    t_hist = np.append(t_hist, t)
                    dp_hist = np.append(
                        dp_hist, p_new.value[0:2].mean() - p_new.value[-2:].mean()
                    )
                    sw_border = np.append(sw_border, sw.value[self.Nx])
                
        self.final_pressure.update_value(p_new)
        self.final_sw.update_value(sw_new)
        return t_hist, rec_fact, sw_border
