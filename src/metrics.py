from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix, BinaryPrecisionRecallCurve, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
import config
import os
import numpy as np
import dataset

class Metrics:
    def __init__(self, device):
        self.device = device
        self.epsilon = 1e-15
        self.metrics = {"F1-score": [], "Confusion_matrix": [], "Precision-Recall curve": [], "mAP":[], "Computation_time_training":[]}
        # Metrics
        if config.NBR_CLASSES==1:
            self.metricF1 = BinaryF1Score()
            self.metricConfusionMatrix = BinaryConfusionMatrix()
            self.metricPrecisionRecallCruve = BinaryPrecisionRecallCurve()
            #self.metricAveragePrecision = MeanAveragePrecision()
        else :
            self.metricF1 = MulticlassF1Score(num_classes=config.NBR_CLASSES)
            self.metricConfusionMatrix = ConfusionMatrix(task='multiclass', num_classes=config.NBR_CLASSES)
            self.metricPrecisionRecallCruve = MulticlassPrecisionRecallCurve(num_classes=config.NBR_CLASSES)
            #self.metricAveragePrecision = MeanAveragePrecision(num_classes=config.NBR_CLASSES)

    def setMetric(self, target, value):
        self.metrics["Computation_time_training"].append(value)

    def addValueToMetrics(self, prediction, original):

        #self.metrics["F1-score"].append(self.metricF1(original, prediction))
        self.metrics["Confusion_matrix"].append(self.metricConfusionMatrix.update(prediction, original))
        #self.metrics["Precision-Recall curve"].append(self.metricPrecisionRecallCruve(original, prediction))
        #self.metrics["mAP"].append(self.metricAveragePrecision(original, prediction))


    def plotMetrics(self, name):
        print('test')

        # plt.figure() # F1 score
        # plt.plot(self.metrics["F1-score"])
        # plt.title("F1 score on Dataset")
        # plt.xlabel("Epoch #")
        # plt.ylabel("F1 score")
        # plt.legend(loc="lower left")
        # plt.savefig(os.path.join(config.PLOT_METRICS, name, "_F1score.png"))

        # plt.figure() # Precision-Recall curve
        # plt.plot(self.metrics["Precision-Recall curve"][-1], score = 'True')
        # plt.title("Precision-Recall curve on Dataset")
        # plt.savefig(os.path.join(config.PLOT_METRICS, name, "_PrecisionRecallcurve.png"))

    def confusionMatrix(self, name, prediction, original, normalize):
        self.metrics["Confusion_matrix"].append(self.metricConfusionMatrix(prediction, original))

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

        path =  os.path.join(config.BASE_OUTPUT, config.ID_SESSION, "ConfusionMatrix.png")
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
        #fig.tight_layout()
        plt.savefig(path)
        plt.close()
