import utils
import config
import dataset
import os

def main():
    utils.logMsg("You activated the option for augmented data.", "info")

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
    print(f"Original images:  {len(imagePaths)} - Original masks: {len(maskPaths)}")

    # Create more data
    if config.GENERATE_AUGMENTED_DATA == True:
        dataset.SegmentationDataset.augment_data(imagePaths, maskPaths, path, augment=True)

if __name__ == "__main__":
	main()
