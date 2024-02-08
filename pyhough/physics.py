import numpy as np

def constants():
    # fundamental constants
    c = 299792458  # speed of light
    h = 6.626e-34  # Planck's constant
    hbar = h / (2 * np.pi)
    ev = 1.602e-19  # electron volt
    Msun = 1.9885e30  # solar mass
    G = 6.67430e-11  # gravitational constant
    f_earth = 1 / 86400  # Earth's rotation frequency
    StellarDay = 23 * 3600 + 56 * 60 + 4.098903691  # length of a stellar day in seconds

    v0 = 0.000766667  # over c
    vesc = 0.00181459  # over c
    rhodm = 0.4e9 / 1e-6  # dark matter energy density (eV/m3)

    omega_earth_orb = 2 * np.pi / (365.25 * 86400)  # Earth's orbital angular frequency
    omega_earth_rot = 2 * np.pi * f_earth  # Earth's rotational angular frequency
    R_earth = 6371e3  # Earth's radius in meters
    Rorb = 149597871e3  # Earth's orbital radius in meters
    eps0 = 8.854187e-12  # vacuum permittivity
    fine_struct = 1 / 137  # fine structure constant

    units = {
        'ev_to_inv_s': ev / hbar,
        'ev_to_inv_m': ev / (hbar * c),
        'ev_to_kg': ev / c**2,
        'kg_to_ev': c**2 / ev,
        'charge_LH': ev / np.sqrt(4 * np.pi * fine_struct),
        'charge_G': ev / np.sqrt(fine_struct),
        'kpc_to_m': 3.086e19,
        'mpc_to_m': 3.086e22
    }

    consts = {
        'c': c,
        'h': h,
        'hbar': hbar,
        'hbar_inev': hbar / ev,
        'ev': ev,
        'Msun': Msun,
        'G': G,
        'eps0': eps0,
        'fine_struct': fine_struct,
        'f_earth': f_earth,
        'omega_earth_rot': omega_earth_rot,
        'omega_earth_orb': omega_earth_orb,
        'v_earth_rot': omega_earth_rot * R_earth,
        'v_earth_orb': omega_earth_orb * Rorb,
        'Rorb': Rorb,
        'Re': R_earth,
        'v0': v0,
        'vesc': vesc,
        'rhodm': rhodm,
        'StellarDay': StellarDay,
        'units': units
    }

    return consts

# Example usage:
# constants_data = constants()
# print(constants_data)

def calc_mc_with_k(k):
    # Assuming constants() is a function that returns the constants dictionary
    consts = constants()
    G = consts['G']
    c = consts['c']
    msun = consts['Msun']

    mc = k**(3/5) * (5 / (96 * np.pi**(8/3)))**(3/5) / (G / c**3)
    mc /= msun

    return mc

