from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix, BinaryPrecisionRecallCurve, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve
import matplotlib.pyplot as plt
import config
import os

class Metrics:
    def __init__(self, device):
        self.device = device
        self.metrics = {"F1-score": [], "Confusion matrix": [], "Precision-Recall curve": [], "mAP":[], "Computation time":[]}
        # Metrics
        if config.NBR_CLASSES==1:
            self.metricF1 = BinaryF1Score().to(self.device)
            self.metricConfusionMatrix = BinaryConfusionMatrix().to(self.device)
            self.metricPrecisionRecallCruve = BinaryPrecisionRecallCurve().to(self.device)
            #self.metricAveragePrecision = MeanAveragePrecision()
        else :
            self.metricF1 = MulticlassF1Score(num_classes=config.NBR_CLASSES).to(self.device)
            self.metricConfusionMatrix = MulticlassConfusionMatrix(num_classes=config.NBR_CLASSES).to(self.device)
            self.metricPrecisionRecallCruve = MulticlassPrecisionRecallCurve(num_classes=config.NBR_CLASSES).to(self.device)
            #self.metricAveragePrecision = MeanAveragePrecision(num_classes=config.NBR_CLASSES)

    def addValueToMetrics(self, original, prediction):
        print(original.shape, prediction.shape)
        self.metrics["F1-score"].append(self.metricF1(original, prediction))
        self.metrics["Confusion matrix"].append(self.metricConfusionMatrix(original, prediction))
        self.metrics["Precision-Recall curve"].append(self.metricPrecisionRecallCruve(original, prediction))
        #self.metrics["mAP"].append(self.metricAveragePrecision(original, prediction))


    def plotMetrics(self, name):

        plt.figure() # F1 score
        plt.plot(self.metrics["F1-score"])
        plt.title("F1 score on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("F1 score")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(config.PLOT_METRICS, name, "_F1score.png"))

        plt.figure() # Confusion matrix
        plt.plot(self.metrics["Confusion matrix"][-1])
        plt.title("Confusion matrix on Dataset")
        plt.savefig(os.path.join(config.PLOT_METRICS, name, "_ConfusionMatrix.png"))

        plt.figure() # Precision-Recall curve
        plt.plot(self.metrics["Precision-Recall curve"][-1], score = 'True')
        plt.title("Precision-Recall curve on Dataset")
        plt.savefig(os.path.join(config.PLOT_METRICS, name, "_PrecisionRecallcurve.png"))
