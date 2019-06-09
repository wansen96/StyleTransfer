dataroot = './dataset'
batch_size = 1
resize_or_crop = 'resize_and_crop' # [resize_and_crop|crop|scale_width|scale_width_and_crop|none]
output_nc = 3
input_nc = 3

# General
model = 'cycle_gan'
no_lsgan = True
gpu_ids = [0] #e.g. 0  0,1,2, 0,2. use -1 for CPU'

# Training
epoch = 'latest'
save_epoch_freq = 10
lambda_A = 10.0
lambda_B = 10.0
save_by_iter = True
niter = 100  # of iter at starting learning rate
niter_decay = 100 # of iter to linearly decay learning rate to zero
lambda_identity = 0.5


# CGAN
lr = 0.0002
b1 = 0.5
b2 = 0.999
channels = 3
out_channels = 3
n_residual_blocks = 9

lambda_identity=0.5
lambda_A=10.
lambda_B=10.
image_size = (256,256)
device = 'cuda'

# name of dataset
set_A = 'man'
set_B = 'boy'
save_dir = f'saved_models_{set_A}_{set_B}/'