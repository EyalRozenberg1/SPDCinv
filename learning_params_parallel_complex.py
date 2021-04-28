
learn_mode           = True  # learn/infer
learn_pump_coeffs    = True  # learn pump coefficients
learn_pump_waists    = True  # learn pump waists
learn_crystal_coeffs = True  # learn crystal coefficients
learn_crystal_waists = True  # learn crystal waists

keep_best      = True

save_stats    = True
show_res      = False  # display results 0/1
save_res      = True  # save results
save_tgt      = False  # save targets

res_path   = 'results/'  # path to results folder
Pt_path    = 'targets/'  # path to targets folder
stats_path = 'stats/'

seed = 1989

"Learning Hyperparameters"
loss_type   = 'sparse_balanced'  # 'kl_sparse_balanced'  # l1:L1 Norm, kl:Kullback Leibler Divergence
step_size   = 0.005
num_epochs  = 256
N           = 1000  # 100, 500, 1000  - number of total-iterations for learning (dataset size)
N_inference = 4000  # 100, 500, 1000  - number of total-iterations for inference (dataset size)
