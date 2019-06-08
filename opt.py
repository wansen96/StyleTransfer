dataroot = './dataset'
batch_size = 1
serial_batches = False
num_threads = 4
max_dataset_size = 10000
dataset_mode = 'unaligned' #[unaligned | aligned | single]
phase = 'train' # train, val, test, etc
resize_or_crop = 'resize_and_crop' # [resize_and_crop|crop|scale_width|scale_width_and_crop|none]
loadSize = 440
fineSize = 396 # crop images to this
output_nc = 3
input_nc = 3
isTrain = True # True or test
direction = 'AtoB'
no_flip = False

# General
model = 'cycle_gan'
no_lsgan = True
gpu_ids = [0] #e.g. 0  0,1,2, 0,2. use -1 for CPU'

# Training
name = 'experiment'
checkpoints_dir = './checkpoints'
suffix = '' # customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}
epoch = 'latest'
load_iter = 0
continue_train = False
save_latest_freq = 5  #frequency of saving the latest results
save_epoch_freq = 10
epoch_count = 1
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
save_dir = "saved_models/"

lambda_identity=0.5
lambda_A=10.
lambda_B=10.
image_size = (256,256)
device = 'cuda'

# Visiualization
display_id = 1 #window id of the web display
display_winsize = 256
display_ncols = 4 # display all images in a single visdom web panel with certain number of images per row
display_server = "http://localhost"
display_env = 'main'
display_port = 8097
update_html_freq = 1000 # frequency of saving training results to html
no_html = True
display_freq = 400;
print_freq = 100;