# Style Transfer
## Description
ECE 285 MLIP final project

## Requirements
- Python 3.7 or above
- PyTorch 
- numpy
- matplotlib


## Code Organization
### [Part 1]
### CycleGAN/
- ```demo.ipynb```: Run a demo for our code. Display the results of Cycle-GAN
- ```training.ipynb```: Run the training of Cycle-GAN model in notebook.
- ```show_results.ipynb```: Run the code to display the training results.(Almost same as demo)
- ```train.py```: The code for training as ```.py``` file. 
- ```test.py```: The code for showing result as ```.py``` file.
- ```cycle_gan.py```: The model of Cycle-GAN.
- ```networks.py```: The code for Generators and Discriminators in GANS.
- ```opt.py```: The code stores options and parameters.
- ```dataloader.py```: The module for creating dataset and dataloader as well as image visualization.

## Dataset
### CycleGAN/dataset

## How To Start(Demo)
### Cycle-GAN
Before running ```demo.ipynb```, check the following: 
- The running environment meets the requirements.
- All python files shown in Code Organization should be included in the same directory as ```demo.ipynb```.
- The folder named after variable ```set_A``` and ```set_B``` in the ```opt.py``` exist in the folder ```dataset```.
- The existence of saved model ```MODEL NAME HERE``` in the folder ```saved_models_DATASETA HERE_DATASETB HERE```.
- Check the GPU resources are available.

Simply run the notebook and you will get the results as shown.
## Results
### 
### Cycle-GAN


## Authors: 
Wansen Zhang, Tongji Luo, Youbin Mo, Yu Shi

## Acknowledgments
