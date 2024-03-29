# import the necessary packages

from torch.utils.data import Dataset
import torchvision
import torch
import csv
import config
import utils
import numpy as np
from tqdm import tqdm
import os
from torchvision.utils import save_image

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
		self.labeledClasses = SegmentationDataset.openColorizedClassesCSV()
		self.sizeLabeledClasses = len(self.labeledClasses)

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def displayClasses(self):
		if config.NBR_CLASSES == 1:
			utils.logMsg("There is one unlabel class.", "data")
		if config.NBR_CLASSES < len(self.labeledClasses):
			utils.logMsg(f"There are {config.NBR_CLASSES} unlabel classes.", "data")

		# Print the classes with their corresponding RGB code
		utils.logMsg(f"There are {len(self.labeledClasses)} unlabel classes:", "data")
		for key, value in self.labeledClasses.items() :
			print(f"Class {key} : RGB = {value}")

	# This folowing method is static because we want to know everywhere in the code how many labeled classes are tagged.
	@staticmethod
	def openColorizedClassesCSV():
		classes = [] # List of the classes
		with open(config.LABEL_PATH) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')

			for row in csv_reader:
				# Discard the first row (title)
				if row[0]=='name' : row = next(csv_reader)
				classes.append([row[0],[int(i) for i in row[1:4]]])
		return dict(classes)

	@staticmethod
	def convertToAdaptedTensorMask(mask, shape):
		mask = mask.numpy()

		maskChanged = np.zeros(shape)

		if config.NBR_CLASSES == 1:
			mask = mask < 0.5
			maskChanged = mask.astype(int)

		else:
			classesGradiant = np.linspace(0, 1, config.NBR_CLASSES + 1)

			for index, value in enumerate(classesGradiant[:-1]):
				maskTmp = np.logical_and(value <= mask, mask < classesGradiant[index+1])
				maskChanged[index, :] = maskTmp.astype(int)

		return torch.tensor(maskChanged, dtype=torch.float32)

	@staticmethod
	def convertToLabeledTensorMask(maskRGB, shape):
		# MaskRGB is in format of PIL.Image
		maskRGB = np.array(maskRGB)
		# We pick up the list of labeledClasses
		labeledClasses = SegmentationDataset.openColorizedClassesCSV()

		maskRGBChanged = np.zeros(shape)

		# We change maskRGB to fit with the shape
		for index, (_, value) in enumerate(labeledClasses.items()):
			# When we have less classes than the csv file we need to stop
			if index >= config.NBR_CLASSES:
				break

			mask = [
						[
							all(pixel == value) for pixel in row
						] for row in maskRGB
					]
			maskRGBChanged[index] = np.array(mask).astype(int)

		# We do not want a dimension for the void class
		if config.NBR_CLASSES < len(labeledClasses):
			maskRGBChanged[0]= np.ones(shape[1:]) - np.sum(maskRGBChanged[1:], axis=0)

		return torch.tensor(maskRGBChanged, dtype=torch.float32)

	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk
		image = torchvision.io.read_image(imagePath, torchvision.io.ImageReadMode.RGB)
		# load the mask from disk in gray scale
		mask = torchvision.io.read_image(self.maskPaths[idx], torchvision.io.ImageReadMode.GRAY)
		# load the mask from disk in RGB scale
		maskRGB = torchvision.io.read_image(self.maskPaths[idx], torchvision.io.ImageReadMode.RGB)

		# check to see if we are applying any transformations
		if self.transforms is not None:

			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)

			# IMPORTANT: We suppose in that case that the transformation ToTensor() si the last of transforms
			transformForMaskRGB = torchvision.transforms.Compose(self.transforms.transforms[:-1])
			maskRGB = transformForMaskRGB(maskRGB)

			# Compute mean and std for images
			mean, std = torch.mean(mask.float(), dim=(1, 2)), torch.std(mask.float(), dim=(1, 2))

			# Add Normalization transformation just for the mask
			normalization = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean, std)])
			mask = normalization(mask)

		# Check the number of class
		if config.ACTIVATE_LABELED_CLASSES == False or (config.NBR_CLASSES == 1 and config.ACTIVATE_LABELED_CLASSES == False):
			# We make the training mask dataset according the right number of class
			mask = SegmentationDataset.convertToAdaptedTensorMask(
				mask=mask,
				shape = [
					config.NBR_CLASSES,
					config.INPUT_IMAGE_HEIGHT,
					config.INPUT_IMAGE_WIDTH
				]
			)

		else:
			# We make the training mask dataset according the labeled csv file for classes
			mask = SegmentationDataset.convertToLabeledTensorMask(
				maskRGB=maskRGB,
				shape=[
					config.NBR_CLASSES,
					config.INPUT_IMAGE_HEIGHT,
					config.INPUT_IMAGE_WIDTH
				]
			)

		# It is important to transform maskRGB with "ToTensor()" after "convertToLabeledTensorMask" because in the case of 24 classes we want an image with pixels within [0,255].
		maskRGB = self.transforms.transforms[-1](maskRGB)

		# return a tuple of the image and its mask
		return (image, mask, maskRGB)

