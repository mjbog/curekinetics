import pytest
import numpy as np
from curekinetics import CureKinetics, GAS_CONSTANT, DSCData
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

freq_factor = 1.e10  # 1/s
activation_energy = 100.  # kJ/mol
reaction_order = 2  # n
autocatalytic_order = 1  # m


@pytest.fixture()
def cure_kinetics_nth():
    cure_kinetics_nth = CureKinetics(freq_factor, activation_energy, reaction_order)
    return cure_kinetics_nth


@pytest.fixture()
def cure_kinetics_auto():
    cure_kinetics_auto = CureKinetics(freq_factor, activation_energy, reaction_order, autocatalytic_order)
    return cure_kinetics_auto


@pytest.fixture()
def cure_kinetics_martin_single():
    # Cure kinetics model from J.L. Martin 2007 PES paper for second autocatalytic
    # function with beta = 15 from TABLE 3
    A2 = 1.183e10
    E2 = 89.4
    n2 = 0.380
    m2 = 1.424
    HR = (1.0 - 0.535) * 361.8
    cure_kinetics_martin_single = CureKinetics(A2, E2, m2, n2, HR)
    return cure_kinetics_martin_single


def test_verify_gas_constant():
    assert (GAS_CONSTANT == 8.314e-3)


def test_can_instantiate_model_nth_order(cure_kinetics_nth):
    assert (cure_kinetics_nth.freq_factor == freq_factor)
    assert (cure_kinetics_nth.act_energy == activation_energy)
    assert (cure_kinetics_nth.reaction_order == reaction_order)
    assert (cure_kinetics_nth.autocatalytic_order == 0)


def test_can_instantiate_model_auto_catalytic(cure_kinetics_auto):
    assert (cure_kinetics_auto.freq_factor == freq_factor)
    assert (cure_kinetics_auto.act_energy == activation_energy)
    assert (cure_kinetics_auto.reaction_order == reaction_order)
    assert (cure_kinetics_auto.autocatalytic_order == autocatalytic_order)


def test_compute_mechanism_function_nth_order(cure_kinetics_nth):
    assert (cure_kinetics_nth.compute_mechanism_function(0.) == 1.0)
    assert (cure_kinetics_nth.compute_mechanism_function(1.0) == 0.0)
    assert (cure_kinetics_nth.compute_mechanism_function(0.5) == 0.25)


def test_mechanism_function_raise_exception_conversion_out_of_bounds(cure_kinetics_nth):
    with pytest.raises(Exception):
        cure_kinetics_nth.compute_mechanism_function(-0.5)
    with pytest.raises(Exception):
        cure_kinetics_nth.compute_mechanism_function(1.5)


def test_raise_exception_float_not_given(cure_kinetics_nth):
    with pytest.raises(Exception):
        cure_kinetics_nth.compute_mechanism_function('x')


def test_compute_mechanism_function_auto(cure_kinetics_auto):
    assert (cure_kinetics_auto.compute_mechanism_function(0.) == 0.0)
    assert (cure_kinetics_auto.compute_mechanism_function(1.0) == 0.0)
    assert (cure_kinetics_auto.compute_mechanism_function(0.5) == 0.125)


def test_compute_arrhenius(cure_kinetics_auto):
    temp = 300.
    k = freq_factor * np.exp(-activation_energy / (GAS_CONSTANT * temp))
    assert (cure_kinetics_auto.compute_arrhenius(temp) == k)


def test_compute_rate_equation_auto(cure_kinetics_auto):
    temp = 300.
    conversion = 0.5

    k = freq_factor * np.exp(-activation_energy / (GAS_CONSTANT * temp))
    fa = cure_kinetics_auto.compute_mechanism_function(conversion)
    rate = k * fa

    assert (cure_kinetics_auto.compute_rate(temp, conversion) == rate)


def test_conversion_update(cure_kinetics_auto):
    temp = 300.
    conversion = 0.5
    delta_time = 1.0

    da_dt = cure_kinetics_auto.compute_rate(temp, conversion)
    updated_conversion = conversion + da_dt * delta_time
    assert (cure_kinetics_auto.update_conversion(temp, conversion, delta_time) == updated_conversion)


def test_compute_conversion_history(cure_kinetics_auto):
    n = 1000
    time = np.array(range(0, n))
    temperature = np.linspace(173., 273. + 250., n)
    conversion_history = cure_kinetics_auto.compute_conversion_history(time, temperature,
                                                                       return_heat_flow=False)
    assert (len(conversion_history) == n)
    assert (conversion_history.max() <= 1.0)
    assert (conversion_history.min() >= 0.)


def test_return_heat_flow_with_conversion_history(cure_kinetics_auto):
    n = 1000
    time = np.array(range(0, n))
    temperature = np.linspace(173., 273. + 250., n)
    conversion_history, heat_flow_history = \
        cure_kinetics_auto.compute_conversion_history(time, temperature,
                                                      return_heat_flow=True)
    assert (len(heat_flow_history) == n)


def test_conversion_history_raises_exception_on_n_mismatch(cure_kinetics_auto):
    n = 1000
    time = np.array(range(0, n))
    temperature = np.array(range(0, n + 1))
    with pytest.raises(Exception):
        cure_kinetics_auto.compute_conversion_history(time, temperature)


def compare_with_tolerance(value, reference, tolerance):
    return abs(value - reference) <= tolerance


def test_conversion_history_matches_data(cure_kinetics_martin_single):
    n = 1000
    time = np.arange(n)

    ramp = 15. # C/min
    temperature = time * ramp / 60. + 273.

    conversion_history, heat_flow_history = \
        cure_kinetics_martin_single.compute_conversion_history(time, temperature,
                                                               return_heat_flow=True)

    assert compare_with_tolerance(conversion_history[-1], 1.0, 1e-5)
    assert compare_with_tolerance(heat_flow_history.max(), 1.32286, 1e-3)
    assert compare_with_tolerance(np.trapz(heat_flow_history, time),
                                  cure_kinetics_martin_single.heat_of_reaction, 1e-2)


def test_compute_history_for_isothermal_case(cure_kinetics_martin_single):
    temperature = 273. + 125.
    dt = 1.  # time steps for analysis

    ck = cure_kinetics_martin_single
    conversion_history, time, heat_flow_history = ck.compute_isothermal_history(temperature, dt)

    # Verify that the full cure is reached and the integral of the heat flow is
    # equal to the heat of reaction
    assert compare_with_tolerance(conversion_history[-1], 1.0, 1e-2)
    assert compare_with_tolerance(np.trapz(heat_flow_history, time),
                                  cure_kinetics_martin_single.heat_of_reaction, 1e-1)


def test_compute_history_for_ramp_rate_case(cure_kinetics_martin_single):
    ramp = 15.
    dt = 1.

    ck = cure_kinetics_martin_single
    conversion_history, time, temp_history, heat_flow_history = ck.compute_ramp_history(ramp, dt)

    # Verify that the full cure is reached and the integral of the heat flow is
    # equal to the heat of reaction
    assert compare_with_tolerance(conversion_history[-1], 1.0, 1e-2)
    assert compare_with_tolerance(np.trapz(heat_flow_history, time),
                                  cure_kinetics_martin_single.heat_of_reaction, 1e-1)

####
@pytest.fixture()
def dsc_data():
    dsc_data = DSCData(dir_path + '/data/structural/3Cpm_a.xlsx')
    return dsc_data


def test_dsc_raise_error_if_file_does_not_exist():
    with pytest.raises(Exception):
        DSCData('sffgafggsgd')


def test_dsc_read_in_sample_mass(dsc_data):
    assert (dsc_data.sample_mass == 13.2)


def test_dsc_store_time_history(dsc_data):
    assert (dsc_data.time[-1] == 93.057 * 60.)


def test_dsc_store_temperature_history(dsc_data):
    assert (dsc_data.temperature[-1] == 298.4119)


def test_dsc_store_normalized_heat_flow(dsc_data):
    assert compare_with_tolerance(dsc_data.heat_flow[-1], -1.54531 / 13.2, 1e-5)


def test_dsc_determine_limits_on_cure_profile(dsc_data):
    assert (dsc_data.data_bounds[0] == 6173 - 55)
    assert (dsc_data.data_bounds[1] == 7384 - 55)


def test_dsc_compute_heat_of_reaction(dsc_data):
    assert compare_with_tolerance(dsc_data.heat_of_reaction, 216.45, 0.1)
