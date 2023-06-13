import argparse
import numpy as np
from model.utils import EasyConfig
from model.CheXExpertNet import CheXExpertNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Weakly supervised lesion localization training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--img', type=str, required=False, help='img to generate heatmap',
                        default="D:\\dataset\\CXR14\\images\\00015676_007.png")
    parser.add_argument('--thr', type=str, required=False, help='threshhold to judge positive sample',
                        default=[0.7484733, 0.9771796, 0.82719517, 0.5494391, 0.9313159, 0.793822, 0.954534, 0.866073,
                                 0.79247904, 0.9652663, 0.99238455, 0.96270514, 0.88407606, 0.9985708])

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    if cfg.mode == 'train':
        CheXExpertNet.train(cfg)
        CheXExpertNet.test(cfg)
    elif cfg.mode == 'test':
        aucIndividual, aucMean = CheXExpertNet.test(cfg)

        for idx, item in enumerate(aucIndividual):
            print(cfg.dataset.class_names[idx], ":", item)

    elif cfg.mode == 'cam':
        CheXExpertNet.cam(cfg)
    elif cfg.mode == 'mult_cam':
        dictPseudoBoxesForModel, dictGtBoxes, dictImg = CheXExpertNet.mult_cam(cfg)
        np.save("./checkpoints/dictPseudoBoxesForModel.npy", dictPseudoBoxesForModel)
        np.save("./checkpoints/dictGtBoxes.npy", dictGtBoxes)
        np.save("./checkpoints/dictImg.npy", dictImg)
    elif cfg.mode == 'region_fusion':
        dictPseudoBoxesForModel = np.load("./checkpoints/dictPseudoBoxesForModel.npy", allow_pickle=True)
        dictGtBoxes = np.load("./checkpoints/dictGtBoxes.npy", allow_pickle=True)
        dictImg = np.load("./checkpoints/dictImg.npy", allow_pickle=True)
        CheXExpertNet.region_fuser(cfg, dictPseudoBoxesForModel, dictGtBoxes, dictImg)
    elif cfg.mode == 'class_fusion':
        CheXExpertNet.class_fuser(cfg)
    elif cfg.mode == "validate":
        print("lesion Localization stage")
        dictPseudoBoxesForModel, dictGtBoxes, dictImg = CheXExpertNet.mult_cam(cfg)
        CheXExpertNet.region_fuser(cfg, dictPseudoBoxesForModel, dictGtBoxes, dictImg)
        print("Disease classification stage")
        CheXExpertNet.class_fuser(cfg)
    else:
        raise ValueError(f"model {cfg.mode} is unknown")