from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix, BinaryPrecisionRecallCurve, MulticlassF1Score, MulticlassPrecisionRecallCurve
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
import config
import os
import numpy as np
import dataset
import utils
import torch

class Metrics:
    def __init__(self, device):
        self.device = device
        self.epsilon = 1e-15
        self.metrics = {"F1-score": [], "Confusion_matrix": [], "Precision-Recall_curve": [], "mAP":[], "Computation_time_training":[]}
        # Metrics
        if config.NBR_CLASSES==1:
            self.metricF1 = BinaryF1Score(num_classes=config.NBR_CLASSES)
            self.metricConfusionMatrix = BinaryConfusionMatrix(num_classes=config.NBR_CLASSES)
            # self.metricPrecisionRecallCruve = BinaryPrecisionRecallCurve()
            # self.metricAveragePrecision = MeanAveragePrecision()
        else :
            self.metricF1 = MulticlassF1Score(num_classes=config.NBR_CLASSES)
            self.metricConfusionMatrix = ConfusionMatrix(task='multiclass', num_classes=config.NBR_CLASSES)
            #self.metricPrecisionRecallCruve = MulticlassPrecisionRecallCurve(num_classes=config.NBR_CLASSES)
            #self.metricAveragePrecision = MeanAveragePrecision()

    def setMetric(self, target, value):
        self.metrics["Computation_time_training"].append(value)

    def addValueToMetrics(self, prediction, original):
        self.metrics["F1-score"].append(self.metricF1(prediction, original))
        #self.metrics["Precision-Recall_curve"].append(self.metricPrecisionRecallCruve(prediction, original))
        #self.metrics["mAP"].append(self.metricAveragePrecision(prediction, original))

    def writeMeanMetrics(self):
        # mean of F1
        meanF1 = np.array(self.metrics["F1-score"])
        meanF1 = np.sum(meanF1) / meanF1.shape[0]

        meanPRC = []
        meanmAP = []

        # # PRC
        # meanPRC = np.array(self.metrics["Precision-Recall_curve"])
        # meanPRC = np.sum(meanPRC) / meanPRC.shape[0]

        # # mAP
        # meanmAP = np.array(self.metrics["mAP"])
        # meanmAP = np.sum(meanmAP) / meanmAP.shape[0]

        meanMetrics = {"F1": meanF1, "Precision-Recall_curve": meanPRC, "mAP": meanmAP}
        path =  os.path.join(config.BASE_OUTPUT, config.ID_SESSION, "Metrics.txt")

        utils.writeFile(meanMetrics, path)

    def confusionMatrix(self, name, prediction, original, normalize):
        self.metrics["Confusion_matrix"].append(self.metricConfusionMatrix(prediction, original))

        if config.ALL_CONFUSION_MATRIX:
            cm = self.metrics["Confusion_matrix"][-1].numpy()

            if normalize:
                cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + self.epsilon)

            path = os.path.join(config.PLOT_METRICS, name + "_ConfusionMatrix.png")
            Metrics.plotConfusionMatrix(cm, "Normalized Confusion Matrix for Image : " + name, normalize, path)

    def meanConfusionMatrix(self):
        meanCm = np.zeros((config.NBR_CLASSES, config.NBR_CLASSES))

        for cm in self.metrics["Confusion_matrix"]:
            cm_numpy = cm.numpy()
            meanCm = meanCm + cm_numpy

        meanCm = meanCm.astype('float') / (meanCm.sum(axis=1)[:, np.newaxis] + self.epsilon)

        path =  os.path.join(config.BASE_OUTPUT, config.ID_SESSION, config.PLOT_METRICS, "ConfusionMatrix.png")
        Metrics.plotConfusionMatrix(meanCm, "Normalized Confusion Matrix for all tested Image", True, path)

    def plotConfusionMatrix(cm, title, normalize, path):
        labeledDic = dataset.SegmentationDataset.openColorizedClassesCSV()
        classes = np.array(list(labeledDic.keys()))
        classes = classes[:config.NBR_CLASSES]

        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        utils.folderExists(config.PLOT_METRICS)
        plt.savefig(path)
        plt.close()

    def diceCoef(self, y_true, y_pred):

        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)

        intersection = torch.sum(y_true_flat * y_pred_flat)
        union = torch.sum(y_true_flat) + torch.sum(y_pred_flat)

        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)

        return dice.cpu().detach().numpy()

    def plotTrainingMetrics(H):
        utils.logMsg("Plotting and saving  the Loss Function...", "info")
        # plot the training loss and the metrics
        plt.style.use("ggplot") # Loss
        fig, ax = plt.subplots(1, 3, figsize=(20,5))
        fig.suptitle("Metrics during the training")

        ax[0].plot(H["train_loss"], label="train_loss")
        ax[0].plot(H["test_loss"], label="test_loss")
        ax[0].set_title("Training Loss on Dataset")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend(loc="lower left")

        ax[1].plot(H["learning_rate"], label="learning_rate")
        ax[1].set_title("Evolution of the Learning Rate")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Learning Rate")

        ax[2].plot(H["train_dice_metric"], label="train dice coef")
        ax[2].plot(H["test_dice_metric"], label="test dice coef")
        ax[2].set_title("Evolution of the Dice coefficient")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Dice coefficient")
        ax[2].legend(loc="lower left")

        plt.savefig(os.path.join(config.BASE_OUTPUT, config.ID_SESSION, "TrainingMetrics.png"))

