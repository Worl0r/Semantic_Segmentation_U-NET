import config
import train
import test
#import metrics
from datetime import datetime
import utils
import model
import numpy as np
import torch
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist

def funcParallelism(rank, world_size):
        transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                (
                    config.INPUT_IMAGE_HEIGHT,
                    config.INPUT_IMAGE_WIDTH
                )),
            torchvision.transforms.ToTensor()
        ])

        # Init the process to parallize
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500', rank=rank, world_size=world_size)

        if config.TYPE_PROCESS == "train":

            # Initialize our UNet model
            unet = model.UNet(
                    encChannels=config.ENC_CHANNELS,
                    decChannels=config.DEC_CHANNELS,
                ).to(rank)

            # Initialize the TrainModel class
            trainModel = train.TrainModel(unet, transform, "metricsClass", device=rank)

            trainModel.setDevice(rank)
            trainModel.trainModel()

        else:
            # load the image paths in our testing file and randomly select 10
            # image paths
            print("[INFO] [TEST] loading up test image paths...")
            imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
            imagePaths = np.random.choice(imagePaths, size=config.SELECTED_IMAGE_TEST)

            unet = torch.load(config.MODEL_PATH).to(rank)

            print("[INFO] [TEST] Creation of the TestModel class...")
            testModel = test.TestModel(unet, transform)

            testModel.makePredictionDataset(imagePaths)


def main():

    # Time
    utils.logMsg("Script stats at: " + str(datetime.now()), "time")
    utils.logMsg("You start a process related with the ID: " + str(config.ID_SESSION), "info")

    # Define and apply transformations
    # IMPORTANT: "ToTensor()" has to be in last position. It is important for SegmentationDataset class.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                (
                    config.INPUT_IMAGE_HEIGHT,
                    config.INPUT_IMAGE_WIDTH
                )),
            torchvision.transforms.ToTensor()
        ])

    # Initialize the Metrics Class
    #metricsClass = metrics.Metrics()

    # Check info about parallelism
    utils.logMsg("The status of cuda is: " + str(torch.cuda.is_available()) + ".", "parallelism")
    utils.logMsg("The status of MPS is: " + str(torch.backends.mps.is_available()) + ".", "parallelism")
    utils.logMsg("There are: " + str(torch.cuda.device_count()) + " devices.", "parallelism")
    utils.logMsg("Parallelism is activated : " + str(config.ACTIVATE_PARALLELISM), "parallelism")

    device = config.DEVICE

    if config.TYPE_PROCESS == "train":
        ##################### TRAIN ####################
         # Save the config to keep track
        utils.saveConfig()

        # Activate Parallelism
        if config.ACTIVATE_PARALLELISM:
            # Start the training
            mp.spawn(funcParallelism, args=(config.NBR_GPU,), nprocs=config.NBR_GPU)

        else:
             # Initialize our UNet model
            unet = model.UNet(
                    encChannels=config.ENC_CHANNELS,
                    decChannels=config.DEC_CHANNELS,
                ).to(device)

            # Initialize the TrainModel class
            trainModel = train.TrainModel(unet, transform, "metricsClass", device=device)

            # Start the training
            trainModel.trainModel()

        utils.logMsg("Training model finished", "info")
        ################################################

    elif config.TYPE_PROCESS == "test":
        ##################### TEST #####################

        print("[INFO] [TEST] load up model...")

        # Activate Parallelism
        if config.ACTIVATE_PARALLELISM:
            # Start the training
            mp.spawn(funcParallelism, args=(config.NBR_GPU,), nprocs=config.NBR_GPU)

        else:
             # load the image paths in our testing file and randomly select 10
            # image paths
            print("[INFO] [TEST] loading up test image paths...")
            imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
            imagePaths = np.random.choice(imagePaths, size=config.SELECTED_IMAGE_TEST)

            # load our model from disk and flash it to the current device
            unet = torch.load(config.MODEL_PATH).to(device)

            print("[INFO] [TEST] Creation of the TestModel class...")
            testModel = test.TestModel(unet, transform)

            testModel.makePredictionDataset(imagePaths)

        print("[INFO] [TEST] Test model finished.")

        ################################################

    else:
        utils.logMsg("Something went wrong about process type in config", "error")

if __name__ == "__main__":
	main()
