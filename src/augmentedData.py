import utils
import config
import torchvision.transforms.functional as TF
import os
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def augment_data(images, masks, save_path, augment=True):
    H = 1024
    W = 1536

    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(images, masks, transform)

    for idx, (image, mask) in tqdm(enumerate(dataset), total=len(dataset)):
        image_name = os.path.splitext(os.path.basename(images[idx]))[0]
        mask_name = os.path.splitext(os.path.basename(masks[idx]))[0]

        if augment:
            # Horizontal Flip
            img_hflip = transforms.functional.hflip(image)
            msk_hflip = transforms.functional.hflip(mask)
            save_image_and_mask(img_hflip, msk_hflip, image_name+ "_1", mask_name + "_1", idx, save_path)

            # Vertical Flip
            img_vflip = transforms.functional.vflip(image)
            msk_vflip = transforms.functional.vflip(mask)
            save_image_and_mask(img_vflip, msk_vflip, image_name + "_2", mask_name + "_2", idx, save_path)

            # Random Crop
            img_crop, msk_crop = random_crop(image, mask, (2 * H // 3, 2 * W // 3))

            save_image_and_mask(img_crop, msk_crop, image_name + "_3", mask_name + "_3", idx, save_path)

        # Original
        save_image_and_mask(image, mask, image_name + "_4", mask_name + "_4", idx, save_path)

def random_crop(image, mask, size):
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
    img_crop = TF.crop(image, i, j, h, w)
    msk_crop = TF.crop(mask, i, j, h, w)
    return img_crop, msk_crop

def save_image_and_mask(image, mask, image_name, mask_name, idx, save_path):
    save_images = [image, mask]
    save_names = [f"image_{image_name}_{idx}.png", f"mask_{mask_name}_{idx}.png"]

    for i, name in zip(save_images, save_names):
        if "image" in name:
            subfolder = "images"
        else:
            subfolder = "masks"

        img_path = os.path.join(save_path, subfolder, name)
        transforms.functional.to_pil_image(i).save(img_path)

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
        #dataset.SegmentationDataset.augment_data(imagePaths, maskPaths, path, augment=True)
        augment_data(imagePaths, maskPaths, path, augment=True)
    else:
        utils.logMsg("You did not activate the config option to create new data.", "info")

    # Load new images
    imagePaths = sorted(list(utils.list_images(os.path.join(path, "images"))))
    maskPaths = sorted(list(utils.list_images(os.path.join(path, "masks"))))
    utils.logMsg(f"Now you have for augmented images:  {len(imagePaths)} and augmented masks: {len(maskPaths)}", "info")

if __name__ == "__main__":
	main()
