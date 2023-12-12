from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix, BinaryPrecisionRecallCurve, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve 
import matplotlib.pyplot as plt
import config

class Metrics:
    def __init__(self):
        self.metrics = {"F1-score": [], "Confusion matrix": [], "Precision-Recall curve": [], "mAP":[], "Computation time":[]}
        # Metrics
        if config.NBR_CLASSES==1:
            self.metricF1 = BinaryF1Score().to(config.DEVICE)
            self.metricConfusionMatrix = BinaryConfusionMatrix().to(config.DEVICE)
            self.metricPrecisionRecallCruve = BinaryPrecisionRecallCurve().to(config.DEVICE)
            #self.metricAveragePrecision = MeanAveragePrecision()
        else :
            self.metricF1 = MulticlassF1Score().to(config.DEVICE)
            self.metricConfusionMatrix = MulticlassConfusionMatrix().to(config.DEVICE)
            self.metricPrecisionRecallCruve = MulticlassPrecisionRecallCurve().to(config.DEVICE)
            #self.metricAveragePrecision = MeanAveragePrecision()

    def setMetric(self, target, value):
        self.metrics[target].append(value)

    def plotMetrics(self):

        plt.figure() # F1 score
        plt.plot(self.metrics["F1-score"])
        plt.title("F1 score on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("F1 score")
        plt.legend(loc="lower left")
        plt.savefig(config.PLOT_PATH + "/F1score.png")

        plt.figure() # Confusion matrix
        plt.plot(self.metrics["Confusion matrix"][-1])
        plt.title("Confusion matrix on Dataset")
        plt.savefig(config.PLOT_PATH + "/ConfusionMatrix.png")

        plt.figure() # Precision-Recall curve
        plt.plot(self.metrics["Precision-Recall curve"][-1], score = 'True')
        plt.title("Precision-Recall curve on Dataset")
        plt.savefig(config.PLOT_PATH + "/PrecisionRecallcurve.png")
