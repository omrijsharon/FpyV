import numpy as np


def calc_height(current_pressure, init_pressure, init_height, temperature_celsius):
    g = 9.80665 # m/s^2
    M = 0.0289644 # kg/mol
    R = 8.31432 # J/(mol*K)
    T = temperature_celsius + 273.15 # K
    return np.log(init_pressure / current_pressure) * (R * T) / (g * M) + init_height

if __name__ == '__main__':
    init_pressure = 1000 # Pa
    init_height = 0 # m
    temperature_celsius = 20 # C
    current_pressure = 1000.0 - 1e-3 # Pa
    print(1000*calc_height(current_pressure, init_pressure, init_height, temperature_celsius))