# USAGE
# import the necessary packages
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from tqdm import tqdm
import matplotlib
import platform
# if platform.system() != 'Linux':
#     matplotlib.use('Agg', force='True')
import matplotlib.pyplot as plt
import torch
import time
import os
import utils
import sys
import config
import dataset
from datetime import datetime
import random

class TrainModel:
	def __init__(self, model, transforms, metrics, device = config.DEVICE):
		# Save the model
		self.model = model

		# Initialize loss function, optimizer and metrics
		self.lossFunc  = BCEWithLogitsLoss()

		# Define the optimizer
		self.optimizer = Adam(model.parameters(), lr=config.INIT_LR)

		self.transforms = transforms
		self.metrics = metrics
		self.device = device
		self.trainDS = []
		self.testDS = []

	def setDevice(self, device):
		self.device = device

	def plotLoss(H):
		utils.logMsg("Plotting and saving  the Loss Function...", "info")
		# plot the training loss and the metrics
		plt.style.use("ggplot") # Loss
		plt.figure()
		plt.plot(H["train_loss"], label="train_loss")
		plt.plot(H["test_loss"], label="test_loss")
		plt.title("Training Loss on Dataset")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss")
		plt.legend(loc="lower left")
		plt.savefig(os.path.join(config.BASE_OUTPUT, config.ID_SESSION, "TrainingLoss.png"))

	@staticmethod
	def plotSampleTraining(input, torchMask, pred, maskRGB, name, index="", epoch=""):
		plots =[]
		plotTitles = []

		# We create a list of dim mask that we want to plot
		plots.append(input[0].cpu().T)
		plotTitles.append("Original Image")
		plots.append(maskRGB[0].T)
		plotTitles.append("Original Mask RGB")

		# Prediction colored mask

		# We plot just one dimension for nbr_classes egual to 1
		if config.NBR_CLASSES == 1:
			plots.append(torchMask[0, 0].cpu().T)
			plotTitles.append("Binary mask")
			plots.append(pred[0, 0].cpu().detach().numpy().T)
			plotTitles.append("Probabilistic Prediction")

		else:

			for i in range(config.VISUALIZATION_DIM):
				plots.append(torchMask[0, i].cpu().T)
				plotTitles.append(f"Segmented Mask dim {i}/{config.NBR_CLASSES - 1}")

			for i in range(config.VISUALIZATION_DIM):
				plots.append(pred[0, i].cpu().detach().numpy().T)
				plotTitles.append(f"Probabilistic Prediction for dim {i}/{config.NBR_CLASSES - 1}")

		utils.prepare_plot(
			plots,
			plotTitles,
			os.path.join(config.PLOT_TRAIN_PATH, name + "_index_" + str(index) + "_batch_" + str(epoch)),
			"Plot training model samples for the first image of each batch",
			mode="train"
		)

	def createDataset(self, imagePaths, maskPaths):
		# partition the data into training and testing splits using 85% of
		# the data for training and the remaining 15% for testing
		utils.logMsg("Train and test paths spliting...", 'info')
		split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, train_size=1-config.TEST_SPLIT, random_state=42)

		# unpack the data split
		(trainImages, testImages) = split[:2]
		(trainMasks, testMasks) = split[2:]

		# We save testImages in a txt file for later
		TrainModel.saveTestPaths(testImages)

		# create the train and test datasets
		utils.logMsg("Segmentation in Dataset...", 'info')
		self.trainDS = dataset.SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
			transforms=self.transforms)
		self.testDS = dataset.SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
			transforms=self.transforms)

		utils.logMsg(f"found {len(self.trainDS)} examples in the training set...", 'info')
		utils.logMsg(f"found {len(self.testDS)} examples in the test set...", 'info')

		# create the training and test data loaders
		utils.logMsg("Creation of Torch DataLoader...", 'info')
		trainLoader = DataLoader(self.trainDS, shuffle=True,
			batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			num_workers=config.NBR_WORKERS)
		testLoader = DataLoader(self.testDS, shuffle=False,
			batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			num_workers=config.NBR_WORKERS)

		return trainLoader, testLoader

	def saveTestPaths(testImages):
		# write the testing image paths to disk so that we can use then
		# when evaluating/testing our model
		utils.logMsg("Saving testing image paths...", 'info')

		f = open(config.TEST_PATHS, "w")
		f.write("\n".join(testImages))
		f.close()

	def earlyStopping(val_loss, best_val_loss, patience, counter):
		# Vérifier si la perte de validation s'améliore
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			counter = 0  # Réinitialiser le compteur si la perte s'améliore

		else:
			counter += 1

		# Arrêt anticipé si la patience est atteinte
		if counter >= patience:
			utils.logMsg("Early stop to prevent from over fitting.", "info")
			return True

		return False

	def batchTraining(self, trainLoader, epoch):
		utils.logMsg(f"Training batch on epoch {epoch+1} done at " + str(datetime.now()) + ".","time")

		totalTrainLoss = 0

		# Loop over the training set
		for index, (input, torchMask, maskRGB) in enumerate(trainLoader):

			# Send the input to the device
			(input, torchMask) = (input.to(self.device), torchMask.to(self.device))

			# Perform a forward pass and calculate the training loss
			pred = self.model(input)

			loss = self.lossFunc(pred, torchMask)

			# We plot some exemple during the training
			if config.MODE_VISUALIZATION:
				TrainModel.plotSampleTraining(input, torchMask, pred, maskRGB, "TrainingPrediction", index, epoch)

			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			# add the loss to the total training loss so far
			totalTrainLoss += loss

		return totalTrainLoss

	def batchTesting(self, testLoader, startTime, H, epoch):
		utils.logMsg(f"Testing batch on epoch {epoch+1} done at " + str(datetime.now()) + ".","time")
		totalTestLoss = 0

		# set the model in evaluation mode
		self.model.eval()

		# Initialiser les variables pour le suivi de l'arrêt anticipé
		best_val_loss = float('inf')
		counter = 0  # Compteur du nombre d'époques sans amélioration

		# loop over the validation set
		for input, torchMask, _ in testLoader:
			# send the input to the device
			(input, torchMask) = (input.to(self.device), torchMask.to(self.device))

			# make the predictions and calculate the validation loss
			pred = self.model(input)

			# TestLoss
			loss = self.lossFunc(pred, torchMask)
			totalTestLoss += loss

			# Set Metrics
			# self.metrics.setMetric("F1-score", self.metrics.metricF1(pred, torchMask))
			# self.metrics.setMetric("Confusion_matrix", self.metrics.metricConfusionMatrix(pred, torchMask))
			# self.metrics.setMetric("Precision-Recall curve", self.metrics.metricPrecisionRecallCruve(pred, torchMask.int()))
			#self.metrics.setMetric("mAP", self.metricAveragePrecision(pred, torchMask))

			if TrainModel.earlyStopping(loss, best_val_loss, config.PATIENCE, counter):
				# display the total time needed to perform the training
				endTime = time.time()
				utils.logMsg("total time taken to train the model: {:.2f}s".format(endTime - startTime), "time")

				# Plot Loss function
				TrainModel.plotLoss(H)

				# Save the model
				utils.folderExists(os.path.join(config.BASE_OUTPUT, config.ID_SESSION))
				torch.save(self.model, config.MODEL_PATH)

				# Stop the script
				sys.exit()

		return totalTestLoss

	def endTestBatch(self, totalTrainLoss, totalTestLoss, trainSteps, testSteps, H, e):
		# Calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgTestLoss = totalTestLoss / testSteps

		# Update our training history
		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

		# Print the model training and validation information
		utils.logMsg("EPOCH: {}/{} at {}".format(e+1 , config.NUM_EPOCHS, datetime.now()), "time")
		print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

		# Save the model
		utils.logMsg(f"We are saving the model at epoch {e+1}.", "info")
		utils.folderExists(os.path.join(config.BASE_OUTPUT, config.ID_SESSION))
		torch.save(self.model, config.MODEL_PATH)

	def checkAugmentedData():
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
		utils.logMsg(f"Original images:  {len(imagePaths)} - Original masks: {len(maskPaths)}", "data")

		# Create more data
		# if config.GENERATE_AUGMENTED_DATA == True:
		# 	dataset.SegmentationDataset.augment_data(imagePaths, maskPaths, path, augment=True)

		# Load new images
		imagePaths = sorted(list(utils.list_images(os.path.join(path, "images"))))
		maskPaths = sorted(list(utils.list_images(os.path.join(path, "masks"))))
		utils.logMsg(f"Augmented images:  {len(imagePaths)} - Augmented masks: {len(maskPaths)}", "data")

		utils.logMsg("You do not have augmented data, please active the option to create them.", "error")
		RuntimeError("You do not have augmented data in the folder: dataset/augmented_data")

		return imagePaths, maskPaths

	def trainModel(self):
		utils.logMsg("Start of model training", "info")

		if config.ACTIVATE_PARALLELISM:
			# We parallelize the model and it works even if there is just a single node
			self.model = DistributedDataParallel(self.model)

		# We can use another special dataset with more data
		if config.AUG_DATA == True:
			# We import new augmented data
			imagePaths, maskPaths = TrainModel.checkAugmentedData()

			# We shuffle them
			shuffledImagePaths = imagePaths.copy()
			shuffledMaskPaths = maskPaths.copy()
			random.shuffle(shuffledImagePaths)
			random.shuffle(shuffledMaskPaths)

			# We crop this dataset accoding the config file
			imagePaths = shuffledImagePaths[:int(config.AUGMENTED_DATA_SPLIT*len(shuffledImagePaths))]
			maskPaths = shuffledMaskPaths[:int(config.AUGMENTED_DATA_SPLIT*len(shuffledMaskPaths))]

		else:
			# load the image and mask filepaths in a sorted manner
			imagePaths = sorted(list(utils.list_images(config.IMAGE_DATASET_PATH)))
			maskPaths = sorted(list(utils.list_images(config.MASK_DATASET_PATH)))

		# Create datasets
		trainLoader, testLoader = TrainModel.createDataset(self, imagePaths, maskPaths)

		# Calculate steps per epoch for training and test set
		trainSteps = len(self.trainDS) // config.BATCH_SIZE
		testSteps = len(self.testDS) // config.BATCH_SIZE
		if testSteps == 0:
			testSteps = 1

		# Initialize a dictionary to store training history
		H = {"train_loss": [], "test_loss": []}

		# loop over epochs
		utils.logMsg("Training the network...", 'info')
		startTime = time.time()

		for e in tqdm(range(config.NUM_EPOCHS)):
			# set the model in training mode
			self.model.train()

			# Compute one batch of training
			totalTrainLoss = TrainModel.batchTraining(self, trainLoader, e)

			# switch off autograd
			with torch.no_grad():

				# Compute one batch of testing
				totalTestLoss = TrainModel.batchTesting(self, testLoader, startTime, H, e)

				TrainModel.endTestBatch(self, totalTrainLoss, totalTestLoss, trainSteps, testSteps, H, e)


		# display the total time needed to perform the training
		endTime = time.time()
		utils.logMsg("Total time taken to train the model: {:.2f}s".format(endTime - startTime), "time")

		# Set the computation time metric
		self.metrics.setMetric("Computation_time_training", endTime - startTime)

		# Plot Loss function
		TrainModel.plotLoss(H)

		# Save the model
		utils.folderExists(os.path.join(config.BASE_OUTPUT, config.ID_SESSION))
		torch.save(self.model, config.MODEL_PATH)

