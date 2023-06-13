from model.utils.Metric import *


def eval_iobb(cfg, dictPseudoBoxesForModel, dictGtBoxes, cam_weight, target_class=None):
    listWeightedBoxes = \
        [[np.zeros(shape=(1, 4)) for i in range(cfg.dataset.num_classes_with_boxes)] for i in range(len(dictGtBoxes))]

    for item in range(len(dictGtBoxes)):
        for nClass in range(cfg.dataset.num_classes_with_boxes):
            if (target_class is None and dictGtBoxes[item][nClass] is not None) \
            or (target_class is not None and target_class == nClass):
                weightsum = 0
                weightedBoxes = np.ones(shape=(1, 4))
                # Weighted boxes fusion
                for modelIdx, dictPseudoBoxes in enumerate(dictPseudoBoxesForModel):
                    for camIdx, cam_name in enumerate(cfg.box_fusion.cam_list):
                        weight = cam_weight[modelIdx][camIdx][nClass]
                        weightedBoxes += dictPseudoBoxes[cam_name][item][nClass] * weight
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
                listIobb[nClass] = np.append(listIobb[nClass], np.max(iobb))

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

    return meanIobb, meanIobbList


class DiscretePsoFitness:
    def __init__(self, n_model, n_cam, n_class, dictGtBoxes, dictPseudoBoxesForModel, cfg):
        self.n_model = n_model
        self.n_cam = n_cam
        self.n_class = n_class
        self.dictGtBoxes = dictGtBoxes
        self.dictPseudoBoxesForModel = dictPseudoBoxesForModel
        self.cfg = cfg

    def __call__(self, x):
        weight = x.reshape((self.n_model, self.n_cam, self.n_class))

        return -1 * eval_iobb(self.cfg, self.dictPseudoBoxesForModel, self.dictGtBoxes, weight)[0]


class PsoFitness:
    def __init__(self, n_model, n_cam, n_class, dictGtBoxes, dictPseudoBoxesForModel, cfg):
        self.n_model = n_model
        self.n_cam = n_cam
        self.n_class = n_class
        self.dictGtBoxes = dictGtBoxes
        self.dictPseudoBoxesForModel = dictPseudoBoxesForModel
        self.cfg = cfg

    def __call__(self, x):
        fitness = np.zeros(shape=(len(x),))
        for idx, item in enumerate(x):
            weight = item.reshape((self.n_model, self.n_cam, self.n_class))
            fitness[idx] = -1 * eval_iobb(self.cfg, self.dictPseudoBoxesForModel, self.dictGtBoxes, weight)[0]

        return fitness