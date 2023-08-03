# Reference for the three-species Lotka-Volterra Model
parameter_names = ["r1", "a1_1", "a1_2", "a1_3", "r2", "a2_1", "a2_2", "a2_3", "r3", "a3_1", "a3_2", "a3_3", "n_u1", "n_u2", "n_u3"]
parameter_values = [3.0, 2.8, 6.0, 2.0, 1.1, 1.8, 0.5, 2.8, 4.0, 3.0, 6.0, 0]

states = ["u1", "u2", "u3"]
IC = [2.,2.,1.]
tspan = (0.0, 10.0)