# Style Transfer
## Description
ECE 285 MLIP final project

This is project is developed by team A++ composed of Wansen Zhang, Tongji Luo, Youbin Mo, Yu Shi

Marvel comic fans are always complaining about how the movie casting ruin their favorite characters while the movie fans argue that the movie characters can fit into the comic universe perfectly. Here we have found a way to tranfer the marvel figures into different styles so that we could end this argument.

<img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/hulk_comic.jpg" width="200" height="300" />    <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/hulk_real.jpg" width="200" height="300" />   <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/hulk_transfer.png" width="200" height="300" />

## Requirements
- Python 3.7 or above
- PyTorch 
- numpy
- matplotlib


## Code Organization
### Neural_Style_transfer/
- ```NeuralStyleTransfer.ipynb```: Run the neural style transfer code to get the results directly.
- ```NNSF.py```: Organized neural style transfer code as ```.py``` file.
- ```download_model.sh```: Run this code to download the VGG pre-trained model(already been added into NeuralStyleTransfer.ipynb).

### CycleGAN/
- ```demo_CycleGAN.ipynb```: Run a demo for our code. Display the results of Cycle-GAN
- ```training.ipynb```: Run the training of Cycle-GAN model in notebook.
- ```show_results.ipynb```: Run the code to display the training results.(Almost same as demo)
- ```train.py```: The code for training as ```.py``` file. 
- ```test.py```: The code for showing result as ```.py``` file.
- ```cycle_gan.py```: The model of Cycle-GAN.
- ```networks.py```: The code for Generators and Discriminators in GANS.
- ```opt.py```: The code stores options and parameters.
- ```dataloader.py```: The module for creating dataset and dataloader as well as image visualization.

## Dataset
### [Part 1]
### CycleGAN/dataset

## How To Start(Demo)
### Neural style transfer
Before running ```demo_NeuralStyleTransfer.ipynb```, check the following: 
- The running environment meets the requirements.
- Check the GPU resources are available.

### Cycle-GAN
Before running ```demo_CycleGAN.ipynb```, check the following: 
- The running environment meets the requirements.
- All python files shown in Code Organization should be included in the same directory as ```demo.ipynb```.
- The folder named after variable ```set_A``` and ```set_B``` in the ```opt.py``` exist in the folder ```dataset```.
- The existence of saved model ```MODEL NAME HERE``` in the folder ```saved_models_DATASETA HERE_DATASETB HERE```.
- Check the GPU resources are available.

Simply run the notebook and you will get the results as shown.
## Results
### Neural style transfer
#### [images]
<img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/hulk_comic.jpg" width="200" height="300" />    <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/hulk_real.jpg" width="200" height="300" />   <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/hulk_transfer.png" width="200" height="300" />

<img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/thor_comic.jpg" width="200" height="300" />    <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/thor_real.jpg" width="200" height="300" />   <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/thor_transfer.png" width="200" height="300" />

<img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/ironman_comic.jpg" width="200" height="300" />    <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/ironman_real.jpg" width="200" height="300" />   <img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/ironman_transfer.png" width="200" height="300" />

#### [losses]
<img src="https://github.com/wansen96/StyleTransfer/blob/master/Sample_results/loss_total.png" width="300" height="200" />

### Cycle-GAN
#### [images]
#### [loss]


## Authors: 
Wansen Zhang, Tongji Luo, Youbin Mo, Yu Shi

## Acknowledgments
