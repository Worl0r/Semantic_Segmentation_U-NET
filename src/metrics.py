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
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from ignite.metrics import mIoU
from ignite.engine import Engine
from ignite.metrics.confusion_matrix import ConfusionMatrix as CMIgnite

class Metrics:
    def __init__(self, device):
        self.device = device
        self.epsilon = 1e-15
        self.metrics = {"miou": [], "dice": [], "roc": [], "ssim": [], "F1-score": [], "Confusion_matrix": [], "Precision-Recall_curve": [], "mAP":[], "Computation_time_training":[]}

        # Metrics
        if config.NBR_CLASSES==1:
            self.metricF1 = BinaryF1Score()
            self.metricConfusionMatrix = BinaryConfusionMatrix()
            self.metricPrecisionRecallCruve = BinaryPrecisionRecallCurve()
            self.metricAveragePrecision = MeanAveragePrecision(iou_type="segm")
            self.default_evaluator = Engine(Metrics.eval_step)
        else :
            self.metricF1 = MulticlassF1Score(num_classes=config.NBR_CLASSES)
            self.metricConfusionMatrix = ConfusionMatrix(task='multiclass', num_classes=config.NBR_CLASSES)
            self.metricPrecisionRecallCruve = MulticlassPrecisionRecallCurve(num_classes=config.NBR_CLASSES)
            #self.metricAveragePrecision = MeanAveragePrecision()

    def setMetric(self, target, value):
        self.metrics["Computation_time_training"].append(value)

    def addValueToMetrics(self, prediction, original, predictionProb, gtProb, name):
        self.metrics["F1-score"].append(self.metricF1(prediction, original))

        # Precision Recall Curve
        self.metricPrecisionRecallCruve.update(torch.from_numpy(predictionProb), gtProb)

        # mAP
        preds = [
                    dict(
                        masks=torch.tensor([prediction.tolist()], dtype=torch.bool),
                        scores=torch.tensor([0.536]),
                        labels=torch.tensor([0]),
                        )
                ]
        target = [
                    dict(
                        masks=torch.tensor([original.tolist()], dtype=torch.bool),
                        labels=torch.tensor([0]),
                    )
                ]
        self.metricAveragePrecision.update(preds, target)

        if config.NBR_CLASSES == 1:
            # ROC
            fpr, tpr, _ = roc_curve(original.flatten(), prediction.flatten(), pos_label=1)
            roc_auc = auc(fpr, tpr)
            self.metrics["roc"].append([np.array(roc_auc), fpr, tpr])

            if config.ALL_METRICS:
                Metrics.plotRocCurve(self, name + "_ROC.png")

            self.metrics["dice"].append(self.diceCoef(self, prediction, original))
            self.metrics["ssim"].append(ssim(prediction.numpy(), original.numpy(), data_range=prediction.numpy().max() - prediction.numpy().min()))

    def plotPrecisionRecallCurve(self, name):
        # Plot Precision Recall Curve
        fig, ax = plt.subplots(figsize=(10, 10))
        self.metricPrecisionRecallCruve.plot(score=True, ax=ax)
        plt.savefig(os.path.join(os.path.sep.join([config.BASE_OUTPUT, config.ID_SESSION, name ])))
        plt.close()

    def plotMAP(self, name):
        # Plot Precision Recall Curve
        fig, ax = plt.subplots(figsize=(10, 10))
        self.metricAveragePrecision.plot(ax=ax)
        plt.savefig(os.path.join(os.path.sep.join([config.BASE_OUTPUT, config.ID_SESSION, name ])))
        plt.close()

    def plotRocCurve(self, name):
        # Plot ROC curve
        plt.figure()
        plt.plot(self.metrics["roc"][-1][1], self.metrics["roc"][-1][2], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % self.metrics["roc"][-1][0])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        utils.folderExists(config.PLOT_METRICS)
        if name == "ROC_For_All_Images.png":
            plt.savefig(os.path.join(os.path.sep.join([config.BASE_OUTPUT, config.ID_SESSION, name])))
        else:
            plt.savefig(os.path.join(config.PLOT_METRICS, name))

    def writeMeanMetrics(self):
        # mean of F1
        meanF1 = np.array(self.metrics["F1-score"])
        meanF1 = np.sum(meanF1) / meanF1.shape[0]

        # Plot Precision Recall Curve
        Metrics.plotPrecisionRecallCurve(self, "PrecisionRecallCurve_For_All_Images.png")

        # mean of mAP
        Metrics.plotMAP(self, "Mean_Average_Precision_For_All_Images.png")

        # Dice
        meanDice = np.array(self.metrics["dice"])
        meanDice = np.sum(meanDice) / meanDice.shape[0]

        # ROC
        meanROC = self.metrics["roc"]
        roc_auc = sum([meanROC[i][0] for i in range(len(meanROC))]) / len(meanROC)
        fpr = sum([meanROC[i][1] for i in range(len(meanROC))]) / len(meanROC)
        tpr = sum([meanROC[i][2] for i in range(len(meanROC))]) / len(meanROC)
        self.metrics["roc"][-1] = [roc_auc, fpr, tpr]
        Metrics.plotRocCurve(self, "ROC_For_All_Images.png")

        # ssim
        meanSsim = np.array(self.metrics["ssim"])
        meanSsim = np.sum(meanSsim) / meanSsim.shape[0]

        meanMetrics = {"ssim": meanSsim, "F1": meanF1, "dice": meanDice, "roc": self.metrics["roc"][-1]}

        path =  os.path.join(config.BASE_OUTPUT, config.ID_SESSION, "Metrics.txt")

        utils.writeFile(meanMetrics, path)

    def display_error_map(true_image, predicted_image, path):
        # Calculate the error map
        error_map = true_image - predicted_image

        # Display the error map
        plt.imshow(error_map, cmap='gray')
        plt.colorbar(label='Error')
        plt.savefig(path)

    ## Confusion Matrix Methods

    def confusionMatrix(self, name, prediction, original, normalize):
        self.metrics["Confusion_matrix"].append(self.metricConfusionMatrix(prediction, original))
        cm = self.metrics["Confusion_matrix"][-1].numpy()

        # set mIoU
        # mIoU(CMIgnite(num_classes=2), ignore_index=0).attach(self.default_evaluator, 'miou')
        # state = self.default_evaluator.run([(1, 2, prediction), original])
        # self.metrics["miou"].append(state.metric['miou'])

        if config.ALL_METRICS:

            path = os.path.join(config.PLOT_METRICS, name + "_ConfusionMatrix.png")

            Metrics.plotCMNormalizedAndNot(self, cm, name, path)


    def meanConfusionMatrix(self):
        meanCm = np.zeros((config.NBR_CLASSES, config.NBR_CLASSES))

        for cm in self.metrics["Confusion_matrix"]:
            cm_numpy = cm.numpy()
            meanCm = meanCm + cm_numpy

        path =  os.path.join(config.BASE_OUTPUT, config.ID_SESSION, "ConfusionMatrix.png")

        Metrics.plotCMNormalizedAndNot(self, meanCm, "Mean Confusion Matrix", path)

    def plotConfusionMatrix(self, cm, title, normalize, path):

        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + self.epsilon)

        labeledDic = dataset.SegmentationDataset.openColorizedClassesCSV()
        classes = np.array(list(labeledDic.keys()))
        classes = classes[:config.NBR_CLASSES+1]

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

        fmt = '.2f' if normalize else 'f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        utils.folderExists(config.PLOT_METRICS)
        if normalize:
            path = path.replace(".png", "_Normalized.png")

        plt.savefig(path)
        plt.close()

    def plotCMNormalizedAndNot(self, cm, name, path):
        np.set_printoptions(precision=2)

        Metrics.plotConfusionMatrix(self, cm, "Confusion matrix, without normalization : " + name, False, path)

        Metrics.plotConfusionMatrix(self, cm, "Normalized Confusion Matrix for Image : " + name, True, path)

    def diceCoef(self, y_true, y_pred):

        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)

        intersection = torch.sum(y_true_flat * y_pred_flat)
        union = torch.sum(y_true_flat) + torch.sum(y_pred_flat)

        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)

        return dice.cpu().detach().numpy()

    def plotTrainingMetrics(H):
        utils.logMsg("Plotting and saving the Loss Function, decis and learning rate...", "info")

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

    #Sum of Least Squared Errors
    def LS(img_mov, img_ref):
        img1 = img_mov.astype('float64')
        img2 = img_ref.astype('float64')
        r = (img1 - img2)**2
        sse = np.sum(r.ravel())
        sse /= r.ravel().shape[0]
        return sse

    #Sum of Absolute Differences
    def SAD(img_mov, img_ref):
        img1 = img_mov.astype('float64')
        img2 = img_ref.astype('float64')
        ab = np.abs(img1 - img2)
        sav = np.sum(ab.ravel())
        sav /= ab.ravel().shape[0]
        return sav

    # Cross Correlation
    def CC(img_mov,img_ref):
        # Vectorized versions of c,d,e
        a = img_mov.astype('float64')
        b = img_ref.astype('float64')

        # Calculating mean values
        AM = np.mean(a)
        BM = np.mean(b)

        c_vect = (a - AM) * (b - BM)
        d_vect = (a - AM) ** 2
        e_vect = (b - BM) ** 2

        # Finally get r using those vectorized versions
        r_out = np.sum(c_vect) / float(np.sqrt(np.sum(d_vect) * np.sum(e_vect)))
        return r_out

    #Mutual Information
    def MI(img_mov,img_ref):
        hgram, x_edges, y_edges = np.histogram2d(img_mov.ravel(), img_ref.ravel(), bins=20)
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    def eval_step(engine, batch):
        return batch
