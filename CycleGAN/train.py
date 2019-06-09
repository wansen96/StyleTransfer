import time
from cycle_gan import cycleGAN 
import opt
import dataloader
import torch
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    dataset_path = opt.dataroot
    # mode is the name of image folders
    img_set = dataloader.GANTransDataset(dataset_path, mode = opt.set_A, image_size = opt.image_size)
    style_set = dataloader.GANTransDataset(dataset_path, mode = opt.set_B, image_size = opt.image_size)
    dataset = dataloader.GANCombinedDataset(img_set, style_set)
    img_size = len(img_set)    # get the number of images in the dataset.
    style_size = len(style_set)
    dataset_size = len(dataset)
    print('dataset size = %d' %dataset_size)
    print('The number of training images = %d' % img_size)
    print('The number of style images = %d' % style_size)

    model = cycleGAN(opt)      # create a model given opt.model and other options  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)
    model.load_networks(opt.epoch)
    checkpoint = model.load_checkpoint(opt.epoch)
    # create dataloader 
    dataset_loader = DataLoader(dataset, batch_size= 1, shuffle=True) # NEW CODE LINE

    start_epoch = checkpoint['current epoch'] + 1
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        loss_list = torch.zeros(8)
        for i, (real_A,real_B) in enumerate(dataset_loader):  # inner loop within one epoch
            model.set_input(real_A.to(device),real_B.to(device))         # unpack data from dataset and apply preprocessing
            model.forward()
            model.optimize_step()
            loss_list += torch.Tensor(model.return_loss())
        loss_list /= (i+1)
        checkpoint['Loss'].append(loss_list)
        checkpoint['current epoch']=epoch

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks('latest')
            model.save_networks(epoch)
            model.save_checkpoint(checkpoint,'latest')
            model.save_checkpoint(checkpoint, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.