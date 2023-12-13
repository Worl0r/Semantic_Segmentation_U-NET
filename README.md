# Acceleration_Material

The goal of this project was to build a Semantic Segmentation Model on the drone dataset (https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset) by implementing a UNet model.
We used the following link as the basis of our architecture : https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/?fbclid=IwAR1N67RjDRDQDR-c7Ih2115m0A2qE7ciVp2aGNzDMZagRdJ-U1ZFFtNFgS0 and we improved it by implementing classes objects, multiclass classification, metrics, parallelism computation.

Here is the architecture of a UNet model :

<img src="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?size=650x400&amp;lossy=2&amp;strip=1&amp;webp=1" alt="" class="wp-image-26078 entered lazyloaded" width="650" height="400" data-lazy-srcset="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?size=130x80&amp;lossy=2&amp;strip=1&amp;webp=1 130w, https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small-300x185.png?lossy=2&amp;strip=1&amp;webp=1 300w, https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?size=390x240&amp;lossy=2&amp;strip=1&amp;webp=1 390w, https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?lossy=2&amp;strip=1&amp;webp=1 500w" data-lazy-sizes="(max-width: 630px) 100vw, 630px" data-lazy-src="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?size=650x400&amp;lossy=2&amp;strip=1&amp;webp=1" data-ll-status="loaded" sizes="(max-width: 630px) 100vw, 630px" srcset="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?size=130x80&amp;lossy=2&amp;strip=1&amp;webp=1 130w, https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small-300x185.png?lossy=2&amp;strip=1&amp;webp=1 300w, https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?size=390x240&amp;lossy=2&amp;strip=1&amp;webp=1 390w, https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/11/unet_small.png?lossy=2&amp;strip=1&amp;webp=1 500w">

### How to get started

Tune the following variables in the *config.py* file :

**Split the dataset**

* The percentage of your dataset that will constitute the test set : *TEST_SPLIT = 0.15*

* The list of gpus : *os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"*

**Vizualization parameters**

* Switch on/off the vizualisation mode (True if you want to print graphs during the training) : *MODE_VISUALIZATION = False*

* The dimension of the window of vizualisation (there are some buts, it needed to avoid multiples of 4) : *VISUALIZATION_DIM = 6*

**Test or Train the model**

* The process you want to perform #value: {"train", "test"} : *TYPE_PROCESS = "train"*
  
* The ID of your process #value: TYPE_PROCESS + "_JJ_MM_YY_part-1" : *ID_SESSION = "train_12_12_23_part-5"*

**Activate Parallelism**

* Swicth on/off the computation on multiple gpus : *ACTIVATE_PARALLELISM = True*

* The number of subprocesses to use for data loading : *NBR_WORKERS = 24*

* The number of gpus : *NBR_GPU = 3*

**Early Stopping**

* Switch on/off the early stopping to halt the model when the performance on the validation set starts to degrade to avoid overfitting : *EARLY_STOPPING_ACTIVATE = False*

* The number of successive epochs to wait before performing the early stopping if the performance on the validation continues to degrade : *PATIENCE = 5*

**Model parameters**

* The channel dimensions of the encoder (note that the first value denotes the number of channels in our input image, and the subsequent numbers gradually double the channel dimension) : *ENC_CHANNELS= (3, 16, 32, 64, 128)*

* The channel dimensions of the decoder (note that the difference here, when compared with the encoder side, is that the channels gradually decrease by a factor of 2 instead of increasing) : *DEC_CHANNELS = (128, 64, 32, 16)*

* The number of classes = *NBR_CLASSES = 24*

* The height of the rescaled images : *INPUT_IMAGE_HEIGHT = 64*

* The width of the rescaled images : *INPUT_IMAGE_WIDTH = 64*

* The batch size : *BATCH_SIZE = 4*

* The number of epochs : *NUM_EPOCHS = 3*

* The learning rate : *INIT_LR = 0.000001*

* The type of the threshold to create the masks from the original images : *THRESHOLD_TYPE = "mean"*

* The number of randomly selected images from the testing set used for the validation : *SELECTED_IMAGE_TEST = 10*

#### Useful command lines
0. Connect to GRICAD

* ssh bigfoot.ciment

1. Activate virtual environement

* source /applis/environments/conda.sh
* source /applis/environments/cuda_env.sh bigfoot  11.2
* conda activate torch

2. Send your script to GRICAD

* oarsub -S ./GricadScript.sh

3. See the status of your script

* oarstat -fj [ID]

4. Delete your script execution

* oardel [ID]

5. Send files

* rsync -avxH data alchantd@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-material-acceleration/alchantd/data

6. Bonus

Install the right version of torch

* pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio===0.12.1 -f https://download.pytorch.org/whl/torch_stable.html

* List of GPU for Gricas: : -p "gpumodel='A100'"  or -p "gpumodel='V100'"  or -p "gpumodel!='T4'"

* To make executable the script bash: chmod +x SingleGPUScript.sh

* See the status: oarstat -u

* Go to interactive mode:
oarsub -l /nodes=1/gpu=1 -p "gpumodel='A100'"  -I --project pr-material-acceleration

### Paths

* Bettik:
cd /bettik/PROJECTS/pr-material-acceleration/login/      # replace login by your login.


