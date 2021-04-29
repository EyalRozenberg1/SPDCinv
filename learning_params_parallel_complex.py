import os

os.environ["JAX_ENABLE_X64"]       = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


learn_mode           = True  # learn/infer
learn_pump_coeffs    = True  # learn pump coefficients
learn_pump_waists    = True  # learn pump waists
learn_crystal_coeffs = True  # learn crystal coefficients
learn_crystal_waists = True  # learn crystal waists


"Learning Hyperparameters"
loss_type   = 'sparse_balanced'  # 'kl_sparse_balanced'  # l1:L1 Norm, kl:Kullback Leibler Divergence
num_epochs  = 200
N           = 1000  # 100, 500, 1000  - number of total-iterations for learning (dataset size)
N_inference = 4000  # 100, 500, 1000  - number of total-iterations for inference (dataset size)


# Optimization methods
optimizer      = 'rmsprop_momentum'  # 'adam', 'sgd', 'adagrad', 'adamax', 'momentum', 'nesterov', 'rmsprop', 'rmsprop_momentum'
keep_best      = True

exp_decay_lr   = True
step_size      = 0.005
decay_steps    = 50
decay_rate     = 0.5


save_stats    = True
show_res      = False  # display results 0/1
save_res      = True  # save results
save_tgt      = False  # save targets


res_path   = 'results/'  # path to results folder
Pt_path    = 'targets/'  # path to targets folder
stats_path = 'stats/'


seed = 1989
