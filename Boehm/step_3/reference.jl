state_names = [
    "STAT5A",
    "STAT5B",
    "pApB",
    "pApA",
    "pBpB",
    "nucpApA",
    "nucpApB",
    "nucpBpB",
]

observable_names = ["pSTAT5A_rel", "pSTAT5B_rel", "rSTAT5A_rel"]

parameter_names = [
    "Epo_degradation_BaF3",
    "k_exp_hetero",
    "k_exp_homo",
    "k_imp_hetero",
    "k_imp_homo",
    "k_phos",
]

noise_parameter_names = [
    "n_pSTAT5A_rel",
    "n_pSTAT5B_rel",
    "n_rSTAT5A_rel"
]

mechanistic_settings = [
    "boehm_fully_known",
    "boehm_missing_interaction",
    "boehm_missing_state",
    "boehm_missing_observable_pSTAT5A_rel"
]
