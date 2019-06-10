import os
import opt
from cycle_gan import cycleGAN
import dataloader
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from torch.utils.data import DataLoader

    
def load_trained_model(photo = 'landscape', style = 'picasso', epo = 'latest'):
    print('Loading photo = ',photo, 'and style = ',style)
    model = cycleGAN()     # create a model given opt.model and other options
    model.save_dir = f'./saved_models_{photo}_{style}/'
    model.load_networks(epo)
    return model
 

def load_dataset(dataset_path = './dataset', photo = 'landscape', style = 'picasso'):
    landscape_set = dataloader.GANTransDataset(dataset_path, mode = photo)
    style_set = dataloader.GANTransDataset(dataset_path, mode = style)
    dataset = dataloader.GANCombinedDataset(landscape_set, style_set)
    dataset_loader = DataLoader(dataset, batch_size= 1, shuffle=False)
    return dataset_loader
    
def torchimshow(image, ax=None):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    
def plot_result(model):
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,8))
    torchimshow(model.real_A[0],ax=axes[0][0])
    axes[0][0].set_title('real image  A')
    torchimshow(model.fake_B[0],ax=axes[0][1])
    axes[0][1].set_title('style transfered to B')
    torchimshow(model.rec_A[0],ax=axes[0][2])
    axes[0][2].set_title('recovered image A')
    torchimshow(model.real_B[0],ax=axes[1][0])
    axes[1][0].set_title('real image B')
    torchimshow(model.fake_A[0],ax=axes[1][1])
    axes[1][1].set_title('style transfered to A')
    torchimshow(model.rec_B[0],ax=axes[1][2])
    axes[1][2].set_title('recovered image B')
    return
    
def training_model(model_path= 'saved_models', dataset_path = './dataset', photo = 'landscape', style = 'picasso'):
    batch_size = 1
    total_iters = 0
    landscape_set = dataloader.GANTransDataset(dataset_path, mode = photo)
    style_set = dataloader.GANTransDataset(dataset_path, mode = style)
    dataset = dataloader.GANCombinedDataset(landscape_set, style_set)
    landscape_size = len(landscape_set)    # get the number of images in the dataset.
    style_size = len(style_set)
    dataset_size = len(dataset)
    print('dataset size = %d' %dataset_size)
    print('The number of training images = %d' % landscape_size)
    print('The number of style images = %d' % style_size)

    model = cycleGAN(opt)      # create a model given opt.model and other options 
    model.save_dir = model_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)
    model.load_networks(opt.epoch)
    checkpoint = model.load_checkpoint(opt.epoch)
    # create dataloader 
    dataset_loader = DataLoader(dataset, batch_size= batch_size, shuffle=True)

    start_epoch = checkpoint['current epoch'] + 1
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        loss_list = torch.zeros(8)
        for i, (real_A, real_B) in enumerate(dataset_loader):  # inner loop within one epoch
            model.set_input(real_A.to(device), real_B.to(device))  # unpack data from dataset and apply preprocessing
            model.forward()
            model.optimize_step()
            loss_list += torch.Tensor(model.return_loss())
        loss_list /= (i + 1)
        checkpoint['Loss'].append(loss_list)
        checkpoint['current epoch'] = epoch

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks('latest')
            model.save_networks(epoch)
            model.save_checkpoint(checkpoint, 'latest')
            model.save_checkpoint(checkpoint, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()