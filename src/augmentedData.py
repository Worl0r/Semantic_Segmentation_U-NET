import utils
import config
import dataset
import os

def main():
    utils.logMsg("You are going to create augmented data...", "info")

    # Create Paths if it needed
    path = os.path.join(config.WORKING_DIRECTORY_PATH, "dataset", "semantic_drone_dataset", "augmented_data")
    utils.folderExists(path)
    pathImage = os.path.join(path, "images")
    pathMask = os.path.join(path, "masks")
    utils.folderExists(pathImage)
    utils.folderExists(pathMask)

    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(utils.list_images(config.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(utils.list_images(config.MASK_DATASET_PATH)))
    utils.logMsg(f"You have for original images:  {len(imagePaths)} and original masks: {len(maskPaths)}", "info")

    # Create more data
    if config.GENERATE_AUGMENTED_DATA == True:
        dataset.SegmentationDataset.augment_data(imagePaths, maskPaths, path, augment=True)
    else:
        utils.logMsg("You did not activate the config option to create new data.", "info")

    # Load new images
    imagePaths = sorted(list(utils.list_images(os.path.join(path, "images"))))
    maskPaths = sorted(list(utils.list_images(os.path.join(path, "masks"))))
    utils.logMsg(f"Now you have for augmented images:  {len(imagePaths)} and augmented masks: {len(maskPaths)}", "info")

if __name__ == "__main__":
	main()
