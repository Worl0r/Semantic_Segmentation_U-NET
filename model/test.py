import dataset
import config
import numpy as np
import torch
import os
import utils
import torchvision

class TestModel:
	def __init__(self, model, transforms):
		self.model = model
		self.transforms = transforms

	def classToColorForPred(pred):
		# We pick up the list of labeled classes in the csv file.
		labeledClasses = dataset.SegmentationDataset.openColorizedClassesCSV()

		colors = list(labeledClasses.values())
		colors = np.array(colors)

		# We use a threshold in the case of a binary classification.
		if config.NBR_CLASSES == 1:
			# We define the type of threshold for a binary classification
			if config.THRESHOLD_TYPE == "mean":
				value = np.mean(pred, axis=None)
			if config.THRESHOLD_TYPE == "median":
				value = np.median(pred, axis=None)

			# We create a mask in function of the threshold
			mask = pred >= value
			# We create a new dimension
			mask = mask[..., np.newaxis]
			# We convert a mask in colored image
			mask = np.where(mask, colors[0], colors[1]).astype(int)
			# Tris transpose is import in order to be consistent about .T for plotting
			return mask.transpose(2, 0, 1)

		else:
			# We take the most probable class for each pixel
			mask = [
						[
							np.argmax(dim) for dim in row
						] for row in pred.T
					]

			return np.array(colors[mask]).T.astype(int)

	def createFilename(imagePath):
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]

		# Select just the filename without the extension
		filenameSplit = filename.split(".")
		imagePaths = sorted(list(utils.list_images(config.MASK_DATASET_PATH)))
		_, extension = os.path.splitext(os.path.join(imagePaths[0]))

		return filenameSplit[-2] + extension

	def openTorchImage(self, imagePath):
		# Load up the image using the path argument
		image = torchvision.io.read_image(imagePath, torchvision.io.ImageReadMode.RGB)
		image = self.transforms(image)

		# make the channel axis to be the leading one, add a batch
		# dimension, and flash it to the current device
		image = torch.unsqueeze(image, dim=0)

		return image.to(config.DEVICE)

	def openTorchGTImage(self, filename):
		# Load the ground-truth segmentation mask in grayscale mode
		# and resize it
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH, filename)
		gtMask = torchvision.io.read_image(groundTruthPath, torchvision.io.ImageReadMode.RGB)
		return  self.transforms(gtMask)

	def plotSampleTraining(input, torchMask, pred, maskRGB, name, index=""):
		plots =[]
		plotTitles = []

		# We create a list of dim mask that we want to plot
		plots.append(input[0].cpu().T)
		plotTitles.append("Original Image")
		plots.append(maskRGB[0].T)
		plotTitles.append("Original Mask RGB")

		# We plot just one dimension for nbr_classes egual to 1
		if config.NBR_CLASSES == 1:
			plots.append(torchMask[0, 0].cpu().T)
			plotTitles.append("Binary mask")
			plots.append(pred[0, 0].cpu().detach().numpy().T)
			plotTitles.append("Probabilistic Prediction")

		else:

			plots.append(torchMask[0, 0].cpu().T)
			plotTitles.append("Segmented Mask Prediction")

			for i in range(config.VISUALIZATION_DIM):
				plots.append(pred[0, i].cpu().detach().numpy().T)
				plotTitles.append(f"Probabilistic Prediction for dim {i}/{config.NBR_CLASSES - 1}")

		# Build and check my folder for plots
		utils.folderExists(config.PLOT_TEST_PATH)

		utils.prepare_plot(
			plots,
			plotTitles,
			path + name + str(index),
			"Plot training model samples for the first image of each batch",
			mode="test"
		)

	def make_predictions(self, imagePath, index):
		# set model to evaluation mode
		self.model.eval()

		# turn off gradient tracking
		with torch.no_grad():

			# Open the image
			image = TestModel.openTorchImage(self, imagePath)

			# Save an original image
			orig = image.clone()

			# Find the filename for the image
			filename = TestModel.createFilename(imagePath)

			# Open the ground-truth related with the image
			gtMask = TestModel.openTorchGTImage(self, filename)

			# Make the prediction, pass the results through the sigmoid
			# function, and convert the result to a NumPy array
			prediction = self.model(image).squeeze()
			prediction = torch.sigmoid(prediction)
			prediction = prediction.cpu().numpy()

			# We change the prediction matrix in colored image instead of class matrix
			colorPred = TestModel.classToColorForPred(prediction)

			TestModel.plotSampleTraining(
				orig,
				torch.tensor(colorPred).unsqueeze(dim=0),
				torch.tensor(prediction).unsqueeze(dim=0),
				gtMask.unsqueeze(dim=0),
				"TestPrediction_",
				index
			)

	def makePredictionDataset(self, imagePaths):
		# iterate over the randomly selected test image paths
		print("[INFO] [TEST] Start of predictions...")

		for index, path in enumerate(imagePaths):
			# make predictions and visualize the results
			TestModel.make_predictions(self, path, index)
			print(f"[INFO] [TEST] Prediction {index+1} on {len(imagePaths)} done.")
