# number of output neurons of the NN
n_out = 8
# number of mechanistic parameters to be learned 
n_mech = 3 + 6

"""
Given states u, parameters p and timepoint t this function outputs the dynamics of a system, i.e. 
the combination of known dynamics and augmented dynamics. 
"""
function ude_dynamics!(du, u, p, st_nn, t; augmented_dynamics=nn_model)
    STAT5A = @inbounds u[1]
    STAT5B = @inbounds u[2]
    pApB = @inbounds u[3]
    pApA = @inbounds u[4]
    pBpB = @inbounds u[5]
    nucpApA = @inbounds u[6]
    nucpApB = @inbounds u[7]
    nucpBpB = @inbounds u[8]

    # static compartment volumnes
    v_cyt = 1.4
    v_nuc = 0.45

    # augmented dynamics 
    u_aug = augmented_dynamics(u, p.ude, st_nn)[1] 

    # physical dynamics
    # physical parameters
    Epo_degradation_BaF3 = inverse_transform(@inbounds p[1]) #p.Epo_degradation_BaF3
    k_exp_hetero = inverse_transform(@inbounds p[2]) #p.k_exp_hetero
    k_exp_homo = inverse_transform(@inbounds p[3]) #p.k_exp_homo
    k_imp_hetero = inverse_transform(@inbounds p[4]) #p.k_imp_hetero
    k_imp_homo = inverse_transform(@inbounds p[5]) #p.k_imp_homo
    k_phos = inverse_transform(@inbounds p[6]) #p.k_phos

    BaF3_Epo = 1.249999999999999999e-07 * exp(-Epo_degradation_BaF3 * t)

    # differential equations
    dphys_1 = (-2.0 * BaF3_Epo * STAT5A.^2 * k_phos - BaF3_Epo * STAT5A * STAT5B * k_phos + 
    2.0/v_cyt*(v_nuc * k_exp_homo * nucpApA) + v_nuc/v_cyt * k_exp_hetero * nucpApB )
    dphys_2 = (- BaF3_Epo * STAT5A * STAT5B * k_phos - 
        2.0 * BaF3_Epo * STAT5B.^2 * k_phos + 
        v_nuc/v_cyt * k_exp_hetero * nucpApB + 
        2.0 * v_nuc/v_cyt * k_exp_homo * nucpBpB )
    dphys_3 = (BaF3_Epo * STAT5A * STAT5B * k_phos - 
        k_imp_hetero * pApB)
    dphys_4 = (BaF3_Epo * STAT5A.^2 * k_phos - 
        k_imp_homo * pApA)
    dphys_5 = (BaF3_Epo * STAT5B.^2 * k_phos - 
        k_imp_homo * pBpB)
    dphys_6 = (v_cyt/v_nuc * k_imp_homo * pApA - 
        k_exp_homo * nucpApA)
    dphys_7 = (v_cyt/v_nuc * k_imp_hetero * pApB - 
        k_exp_hetero * nucpApB)
    dphys_8 = (v_cyt / v_nuc * k_imp_homo * pBpB - 
        k_exp_homo * nucpBpB)
    
    # combined dynamics
    @inbounds du[1] = u_aug[1] + dphys_1
    @inbounds du[2] = u_aug[2] + dphys_2
    @inbounds du[3] = u_aug[3] + dphys_3
    @inbounds du[4] = u_aug[4] + dphys_4
    @inbounds du[5] = u_aug[5] + dphys_5
    @inbounds du[6] = u_aug[6] + dphys_6
    @inbounds du[7] = u_aug[7] + dphys_7
    @inbounds du[8] = u_aug[8] + dphys_8
end;


"""
Mapping hidden states to observables
"""
function observable_mapping(state, p, st_nn; augmented_dynamics=nn_model)
    specC17 = 0.107

    STAT5A = @inbounds state[1, :];
    STAT5B = @inbounds state[2, :];
    pApB = @inbounds state[3, :];
    pApA = @inbounds state[4, :];
    pBpB = @inbounds state[5, :];

    pSTAT5A_rel = (100.0 .* pApB .+ 200.0 .* pApA .* specC17) ./ (pApB .+ STAT5A .* specC17 .+ 2.0 .* pApA .* specC17)
    pSTAT5B_rel = -(100.0 .* pApB .- 200.0 .* pBpB .* (specC17 .- 1.0)) ./ ((STAT5B .* (specC17 .- 1.0) .- pApB) .+ 2.0 .* pBpB .* (specC17 .- 1.0))
    rSTAT5A_rel = (100.0 .* pApB .+ 100.0 .* STAT5A .* specC17 .+ 200.0 .* pApA .* specC17) ./ (2.0 .* pApB .+ STAT5A .* specC17 .+ 2.0 .* pApA .* specC17 .- STAT5B .* (specC17 .- 1.0) .- 2.0 .* pBpB .* (specC17 .- 1.0))

    return vcat(transpose(pSTAT5A_rel), transpose(pSTAT5B_rel), transpose(rSTAT5A_rel))
end

