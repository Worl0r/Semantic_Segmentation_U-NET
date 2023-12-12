# Acceleration_Material

### How to get started

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
