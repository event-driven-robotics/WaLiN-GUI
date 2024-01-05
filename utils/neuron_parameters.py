"""
Example structure of parameter list:
name = {
    "param_1": [min, max, step, init]
    ...
    "param_n": [min, max, step, init]
}
"""


# Mihilas-Niebur neuron
mn_parameter = {
    "a": [-100, 40, 0.1, 0],
    "A1": [-5, 15, 0.1, 0],
    "A2": [-1, 1, 0.01, 0],
    "b": [-20, 20, 0.1, 10],  # 1/s
    "G": [0, 75, 0.1, 50],  # 1/s
    "k1": [0, 300, 1, 200],  # 1/s
    "k2": [0, 30, 0.1, 20],  # 1/s
    "R1": [0, 2, 0.01, 0],  # Ohm
    "R2": [0, 2, 0.01, 1],  # Ohm
}

# Iziekevich neuron
iz_parameter = {
    "a": [-10, 10, 0.5, 0],
    "b": [-10, 10, 0.5, 0],
    "d": [-10, 10, 0.5, 0],
    "tau": [0.1, 2.0, 0.01, 0],
    "k": [0.0, 300, 10, 0],
}

# Leaky-integrate and firing neuron
lif_parameter = {
    "beta": [0.5, 1.0, 0.01, 0.9],
    "R": [0.1, 10.0, 0.1, 4.0],
    "threshold": [0.1, 1.0, 0.1, 0.1],
}

# Recurrent leaky-integrate and firing neuron
rlif_parameter = {
    "alpha": [0.5, 1.0, 0.01, 0.9],
    "beta": [0.5, 1.0, 0.01, 0.9],
    "R": [0.1, 10.0, 0.1, 4.0],
    "threshold": [0.1, 1.0, 0.1, 0.1],
}
