n_out = 3

function ude_dynamics!(du, u, p, st_nn, t; nn_model)
    #parameter log transform 
    r1 = inverse_transform(p[1])
    a1_1 = inverse_transform(p[2])
    a1_2 = inverse_transform(p[3])
    a1_3 = inverse_transform(p[4])
    r2 = inverse_transform(p[5])
    a2_1 = inverse_transform(p[6])
    a2_2 = inverse_transform(p[7])
    a2_3 = inverse_transform(p[8])
    r3 = inverse_transform(p[9])
    a3_1 = inverse_transform(p[10])
    a3_2 = inverse_transform(p[11])
    a3_3 = inverse_transform(p[12])
    
    #create the neural network for the dynamics 
    nn_dynamics = nn_model(u,p.ude,st_nn)[1]

    #the physical dynamics
    dphys1 = -r1*u[1] + a1_3*u[1]*u[3] + a1_2*u[1]*u[2] - a1_1*u[1]*u[1]
    dphys2 = -r2*u[2] + a2_3*u[2]*u[3] - a2_1*u[2]*u[1] - a2_2*u[2]*u[2]
    dphys3 = r3*u[3] - a3_2*u[3]*u[2] - a3_1*u[3]*u[1] - a3_3*u[3]*u[3]
    
    #the physical dynamics with nn dynamics aiming to be 0 
    du[1] = dphys1 + nn_dynamics[1]
    du[2] = dphys2 + nn_dynamics[2]
    du[3] = dphys3 + nn_dynamics[3]
end