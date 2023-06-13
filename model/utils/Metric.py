import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


def computeAUCROC (dataGT, dataPRED, classCount):
    """
    Parameters
    ----------
    dataGT:   (N, C) ndarray of float  
    dataPRED: (N, C) ndarray of float  
    Returns:  AUC, ACC under the best thresholds, Best Thresholds
    -------
    """
    # --- caculate AUC
    outAUC = []

    if isinstance(dataGT, torch.Tensor):
        datanpGT = dataGT.cpu().numpy()
    else:
        datanpGT = dataGT

    if isinstance(dataPRED, torch.Tensor):
        datanpPRED = torch.sigmoid(dataPRED).cpu().numpy()
    else:
        datanpPRED = dataPRED


    for i in range(classCount):
        outAUC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    # --- plot roc curve
    plt.figure(figsize=(20, 10))
    for i in range(classCount):
        fpr, tpr, thresholds = roc_curve(datanpGT[:, i], datanpPRED[:, i])
        plt.subplot(3, 5, i + 1)
        plt.title(str(i), pad=20)
        plt.plot(fpr, tpr, label='ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')

    # --- caculate the best threshlod by F1 score
    outBestThresholds = []
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(classCount):
        precision, recall, thresholds = precision_recall_curve(datanpGT[:, i], datanpPRED[:, i])
        f1 = np.multiply(2, np.divide(np.multiply(precision, recall), np.add(precision, recall)))
        outBestThresholds.append(thresholds[np.where(f1 == max(f1))][0])

    plt.subplot(3, 5, classCount + 1)
    plt.title("best_thresholds", pad=20)
    plt.plot(outBestThresholds, label='threshold')
    plt.xlabel('class')
    plt.ylabel('threshold')
    plt.show()

    # --- caculate the acc under the best threshlod
    outACC = []
    for i in range(classCount):

        tp = len(np.where((datanpPRED[:, i] > outBestThresholds[i]) == datanpGT[:, i])[0])
        outACC.append(tp /datanpPRED[:, i].size)

    return outAUC, outACC, outBestThresholds


def computeIoU(bboxGT, bboxPRED):
    """
    Parameters
    ----------
    bboxPRED: (N, 4) ndarray of float  [x1,y1,x2,y2]
    bboxGT: (K, 4) ndarray of float  [x1,y1,x2,y2]
    Returns: (N, K) ndarray of IoU between boxes and query_boxes
    -------
    """
    area = (bboxGT[:, 2] - bboxGT[:, 0]) * (bboxGT[:, 3] - bboxGT[:, 1])

    iw = np.minimum(np.expand_dims(bboxPRED[:, 2], axis=1), bboxGT[:, 2]) - np.maximum(np.expand_dims(bboxPRED[:, 0], 1), bboxGT[:, 0])
    ih = np.minimum(np.expand_dims(bboxPRED[:, 3], axis=1), bboxGT[:, 3]) - np.maximum(np.expand_dims(bboxPRED[:, 1], 1), bboxGT[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((bboxPRED[:, 2] - bboxPRED[:, 0]) * (bboxPRED[:, 3] - bboxPRED[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def computeIoBB(bboxGT, bboxPRED):
    """
    IoBB a.k.a IoR!
    Parameters
    ----------
    bboxPRED: (N, 4) ndarray of float  [x1,y1,x2,y2]
    bboxGT: (K, 4) ndarray of float  [x1,y1,x2,y2]
    Returns: (N, K) ndarray of IoBB between boxes and query_boxes
    -------
    """
    if len(bboxPRED) == 0:
        return 0.0

    areaPRED = np.max((bboxPRED[:, 2] - bboxPRED[:, 0]) * (bboxPRED[:, 3] - bboxPRED[:, 1]))
    areaGT = np.max((bboxGT[:, 2] - bboxGT[:, 0]) * (bboxGT[:, 3] - bboxGT[:, 1]))
    area = areaPRED if areaGT > areaPRED else areaGT

    iw = np.minimum(np.expand_dims(bboxPRED[:, 2], axis=1), bboxGT[:, 2]) - np.maximum(np.expand_dims(bboxPRED[:, 0], 1), bboxGT[:, 0])
    ih = np.minimum(np.expand_dims(bboxPRED[:, 3], axis=1), bboxGT[:, 3]) - np.maximum(np.expand_dims(bboxPRED[:, 1], 1), bboxGT[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    intersection = np.squeeze(iw * ih)

    if areaPRED != 0:
        return intersection / areaPRED
    else:
        return 0


def computePLA(bboxGT, bboxPRED):
    """
    Parameters
    ----------
    bboxPRED: (N, 4) ndarray of float  [x1,y1,x2,y2]
    bboxGT: (K, 4) ndarray of float  [x1,y1,x2,y2]
    Returns: (N, K) ndarray of IoBB between boxes and query_boxes
    -------
    """

    if len(bboxPRED) == 0:
        return 0.0

    areaGT = np.max((bboxGT[:, 2] - bboxGT[:, 0]) * (bboxGT[:, 3] - bboxGT[:, 1]))

    iw = np.minimum(np.expand_dims(bboxPRED[:, 2], axis=1), bboxGT[:, 2]) - np.maximum(np.expand_dims(bboxPRED[:, 0], 1), bboxGT[:, 0])
    ih = np.minimum(np.expand_dims(bboxPRED[:, 3], axis=1), bboxGT[:, 3]) - np.maximum(np.expand_dims(bboxPRED[:, 1], 1), bboxGT[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    intersection = np.squeeze(iw * ih)

    return np.mean(intersection) > 0.0
