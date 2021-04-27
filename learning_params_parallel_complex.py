
learn_mode    = True  # learn/infer
save_stats    = True
show_res      = True  # display results 0/1
save_res      = True  # save results
save_tgt      = True  # save targets
learn_crystal = True  # learn crystal or pump only

res_path   = 'results/'  # path to results folder
Pt_path    = 'targets/'  # path to targets folder
stats_path = 'stats/'

seed = 1989

"Learning Hyperparameters"
loss_type   = 'kl_sparse_balanced'  # 'kl_sparse_balanced'  # l1:L1 Norm, kl:Kullback Leibler Divergence
step_size   = 0.05
num_epochs  = 100
N           = 800  # 100, 500, 1000  - number of total-iterations for learning (dataset size)
N_inference = 4000  # 100, 500, 1000  - number of total-iterations for inference (dataset size)
