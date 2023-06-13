from model.utils.Metric import *
import matplotlib.pyplot as plt


def eval_iobb(cfg, dictPseudoBoxesForModel, dictGtBoxes, cam_weight, target_class=None, thr=None, debug=False):

    listWeightedBoxes = \
        [[np.zeros(shape=(1, 4)) for i in range(cfg.dataset.num_classes_with_boxes)] for i in range(len(dictGtBoxes))]

    for item in range(len(dictGtBoxes)):
        for nClass in range(cfg.dataset.num_classes_with_boxes):
            if (target_class is None and dictGtBoxes[item][nClass] is not None) \
            or (target_class is not None and target_class == nClass):

                weightsum = 0
                weightedBoxes = np.zeros(shape=(1, 4))
                # Weighted boxes fusion
                for modelIdx, dictPseudoBoxes in enumerate(dictPseudoBoxesForModel):
                    for camIdx, cam_name in enumerate(cfg.box_fusion.cam_list):
                        # Gain weight
                        if target_class is None:
                            weight = cam_weight[nClass][modelIdx*len(cfg.box_fusion.cam_list) + camIdx]
                        else:
                            weight = cam_weight[modelIdx][camIdx]
                        # Fuse boxes
                        pseudoBoxes = dictPseudoBoxes[cam_name][item][nClass]
                        pseudoBox = None

                        # If there are multiple boxes, select the box with the highest iobb
                        if pseudoBoxes.shape[0] > 1:

                            if debug:
                                fig, ax = plt.subplots()
                                ax.set_xlim(0, 512)
                                ax.set_ylim(0, 512)
                                gtBox = dictGtBoxes[item][nClass]
                                ax.add_patch(plt.Rectangle(
                                    (gtBox[0][0], gtBox[0][1]), gtBox[0][2] - gtBox[0][0],
                                                                gtBox[0][3] - gtBox[0][1],
                                    color="red", fill=False, linewidth=2))

                            maxIobb = -1
                            for box in pseudoBoxes:
                                box = box.reshape(1,4)
                                iobb = computeIoBB(dictGtBoxes[item][nClass], box)
                                if iobb > maxIobb:
                                    pseudoBox = box
                                    maxIobb = iobb

                                if debug:
                                    ax.add_patch(plt.Rectangle(
                                        (box[0][0], box[0][1]), box[0][2] - box[0][0], box[0][3] - box[0][1],
                                        color="black", fill=False, linewidth=2))
                                    # plt.show()
                            if debug:
                                plt.savefig(f"./heatmaps/{item}_{nClass}")
                                plt.close()

                        elif len(pseudoBoxes) > 0:
                                pseudoBox = pseudoBoxes[0]

                        if pseudoBox is None or pseudoBox.size == 0:
                            pseudoBox = np.zeros((1, 4))

                        weightedBoxes += pseudoBox * weight
                        weightsum += weight
                if weightsum != 0:
                    weightedBoxes /= weightsum

                listWeightedBoxes[item][nClass] = weightedBoxes

    # calculate Iobb
    listIobb = [np.array([])] * cfg.dataset.num_classes_with_boxes
    for item in range(len(dictGtBoxes)):
        for nClass in range(cfg.dataset.num_classes_with_boxes):
            gtBox = dictGtBoxes[item][nClass]
            WeightedBox = listWeightedBoxes[item][nClass]
            if gtBox is not None and WeightedBox is not None and np.sum(WeightedBox) != 0:
                iobb = computeIoBB(gtBox, WeightedBox)
                iobb = np.max(iobb)

                if thr is not None:
                    iobb = 1.0 if iobb > thr else 0.0

                listIobb[nClass] = np.append(listIobb[nClass], iobb)

    # set Nan to 0
    listIobb = np.nan_to_num(np.array(listIobb, dtype=object))

    if target_class is None:
        # print("iobb:")
        meanIobb = 0
        meanIobbList = []
        for n in range(cfg.dataset.num_classes_with_boxes):
            mean = np.mean(listIobb[n])
            # print(cfg.dataset.class_names[n] + ": " + str(mean))
            meanIobb = meanIobb + mean
            meanIobbList.append(mean)

        meanIobb /= cfg.dataset.num_classes_with_boxes
        # print("meanIobb: ", meanIobb)

        return meanIobb, meanIobbList, listIobb, listWeightedBoxes
    else:
        if len(listIobb[target_class]) > 0 :
            return np.mean(listIobb[target_class]), None
        else:
            return 0, None, None, None


class DiscretePsoFitness:
    def __init__(self, n_model, n_cam, target_class, dictGtBoxes, dictPseudoBoxesForModel, cfg):
        self.n_model = n_model
        self.n_cam = n_cam
        self.target_class = target_class
        self.dictGtBoxes = dictGtBoxes
        self.dictPseudoBoxesForModel = dictPseudoBoxesForModel
        self.cfg = cfg

    def __call__(self, x):
        weight = x.reshape((self.n_model, self.n_cam))
        meanIobb = eval_iobb(self.cfg, self.dictPseudoBoxesForModel, self.dictGtBoxes, weight, self.target_class)[0]
        return -1 * meanIobb


class PsoFitness:
    def __init__(self, n_model, n_cam, target_class, dictGtBoxes, dictPseudoBoxesForModel, cfg):
        self.n_model = n_model
        self.n_cam = n_cam
        self.target_class = target_class
        self.dictGtBoxes = dictGtBoxes
        self.dictPseudoBoxesForModel = dictPseudoBoxesForModel
        self.cfg = cfg

    def __call__(self, x):
        fitness = np.zeros(shape=(len(x),))
        for idx, item in enumerate(x):
            weight = item.reshape((self.n_model, self.n_cam))
            meanIobb = eval_iobb(self.cfg, self.dictPseudoBoxesForModel, self.dictGtBoxes, weight, self.target_class)[0]
            fitness[idx] = -1 * meanIobb
        return fitness