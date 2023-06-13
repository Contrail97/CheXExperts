import numpy as np
from model.regionFuser.Evaluation import eval_iobb
from model.regionFuser.Pso import pso, discretePso, binaryPso

class RegionFuser:
    def __init__(self, cfg):
        self.cfg = cfg

    def optimize(self, dictPseudoBoxesForModel, dictGtBoxes):

        weight_list = []
        iobb_list = []

        if self.cfg.box_fusion.optimizer == "pso":
            return pso(self.cfg, dictPseudoBoxesForModel, dictGtBoxes)

        elif self.cfg.box_fusion.optimizer == "discretePso":

            for nClass in range(self.cfg.dataset.num_classes_with_boxes):
                print("Optimization Objective: ", self.cfg.dataset.class_names[nClass])
                class_mean_Iobb, class_weight = discretePso(self.cfg, dictPseudoBoxesForModel, dictGtBoxes, nClass)
                weight_list.append(class_weight)
                iobb_list.append(class_mean_Iobb)

            for idx, item in enumerate(iobb_list):
                print(self.cfg.dataset.class_names[idx] + ": " + str(item))

            mean_iobb = np.mean(iobb_list)
            print("mean iobb:", mean_iobb)

            return mean_iobb, np.array(weight_list)

        elif self.cfg.box_fusion.optimizer == "binaryPso":

            for nClass in range(self.cfg.dataset.num_classes_with_boxes):
                print("Optimization Objective: ", self.cfg.dataset.class_names[nClass])
                class_mean_Iobb, class_weight = binaryPso(self.cfg, dictPseudoBoxesForModel, dictGtBoxes, nClass)
                weight_list.append(class_weight)
                iobb_list.append(class_mean_Iobb)

            for idx, item in enumerate(iobb_list):
                print(self.cfg.dataset.class_names[idx] + ": " + str(item))

            mean_iobb = np.mean(iobb_list)
            print("mean iobb:", mean_iobb)

            return mean_iobb, np.array(weight_list)

        else:
            raise Exception('Unknown optimizer : {}'.format(self.cfg.boxfuser.optimizer))

    def eval(self, dictPseudoBoxesForModel, dictGtBoxes, cam_weight, thr=None):
        return eval_iobb(self.cfg, dictPseudoBoxesForModel, dictGtBoxes, cam_weight, thr=thr, debug=True)
