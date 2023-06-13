import os
import gc
import sys
import cv2
import time
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from model.Losses import Losses
from model.regionFuser.RegionFuser import RegionFuser
from torch.cuda.amp import autocast, GradScaler
from model.ChexClassifier import ChexClassifier
from model.utils.Logger import build_logger
from model.utils.Wandb import WandbRecoder
from model.utils.Metric import computeAUCROC, computePLA, computeIoBB
from model.dataset.DataTransforms import DataTransforms
from model.dataset.DatasetGenerator import DatasetGenerator, BBoxDatasetGenerator, BBoxCollater


class CheXExpertNet(object):

    currEpoch = 0
    startTime = 0
    logger = None

    @staticmethod
    def train(cfg):
        # set mode
        cfg.mode = "train"
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True

        # set random seed
        nSeed = 1662383712753
        random.seed(nSeed)
        torch.manual_seed(nSeed)
        torch.cuda.manual_seed(nSeed)
        torch.cuda.manual_seed_all(nSeed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # network architecture
        model = ChexClassifier(cfg).cuda()

        # data transforms
        transSeqTrain = DataTransforms(cfg).compile("train")
        transSeqVal = DataTransforms(cfg).compile("val")

        # dataset builders
        datasetTrain = DatasetGenerator(pathImageDirectory=[cfg.dataset.images_path, cfg.dataset.segment_path],
                                        pathDatasetFile=cfg.dataset.file_train, transform=transSeqTrain)
        datasetVal = DatasetGenerator(pathImageDirectory=[cfg.dataset.images_path, cfg.dataset.segment_path],
                                      pathDatasetFile=cfg.dataset.file_val, transform=transSeqVal)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=cfg.train.batch_size,
                                     shuffle=True,  num_workers=cfg.dataset.num_workers, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=cfg.val_test.batch_size,
                                   shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

        datasetBBoxTest = BBoxDatasetGenerator(pathImageDirectory=cfg.dataset.images_path,
                                           pathDatasetFile=cfg.dataset.file_bbox,
                                           transform=transSeqVal)
        dataLoaderBBoxTest = DataLoader(dataset=datasetBBoxTest, batch_size=cfg.val_test.batch_size, shuffle=False,
                                    num_workers=cfg.dataset.num_workers, collate_fn=BBoxCollater)

        # optimizer & scheduler
        optimizer = CheXExpertNet.build_optimizer_from_cfg(model.parameters(), cfg)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
        scaler = GradScaler()

        # loss
        loss = CheXExpertNet.build_loss_from_cfg(cfg)

        # load checkpoint
        epochStart = CheXExpertNet.load_checkpoint_from_cfg(cfg, model, optimizer)

        # set logger
        CheXExpertNet.logger = CheXExpertNet.build_logger_from_cfg(cfg)
        CheXExpertNet.logger.info(cfg)

        # set wandb
        wandbRecoder = WandbRecoder(cfg)

        # set random seed
        # nSeed = 1662383712753.199
        # random.seed(nSeed)
        # CheXExpertNet.logger.info("Seed is " + str(nSeed))

        # train the network
        maxAUC = 0.0
        maxIoBB = 0.0
        CheXExpertNet.startTime = time.time()

        for epochID in range(epochStart, cfg.train.max_epoch):
            CheXExpertNet.currEpoch = epochID
            CheXExpertNet.epoch_train(model, dataLoaderTrain, optimizer, loss, scaler, cfg)
            lossVal, lossMean, aucVal, aucMean, accVal, accMean, bThr = CheXExpertNet.epoch_val(model, dataLoaderVal, loss, cfg)
            listIoBB, iobbMean = CheXExpertNet.localization(cfg, model, dataLoaderBBoxTest)
            scheduler.step(lossMean)

            # save checkpoints
            pthDict = \
                {'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_auc': maxAUC, 'optimizer':optimizer.state_dict()}
            if aucMean > maxAUC:
                maxAUC = aucMean
                torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'best_auc_model' + str(epochID + 1) + '-' + str(maxAUC) + '.pth.tar'))
                torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'best_auc_model.pth.tar'))
            if iobbMean > maxIoBB:
                maxIoBB = iobbMean
                torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'best_iobb_model' + str(epochID + 1) + '-' + str(maxIoBB) + '.pth.tar'))
                torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'best_iobb_model.pth.tar'))
            torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'newest_model.pth.tar'))

            # log
            summary = "Epoch:{} | meanLoss:{} | meanAcc:{}  meanAuc:{} bestAUC:{} | meanIoBB:{} bestIoBB:{} \n\t  AUC :{} \n\t  ACC :{} \n\t  LOSS:{} \n\t bThr:{} \n\t iobb:{} \n\t"\
                .format(epochID + 1, lossMean, accMean, aucMean, maxAUC, iobbMean, maxIoBB, aucVal, accVal, lossVal, bThr, listIoBB.tolist())
            CheXExpertNet.logger.info(summary)

            # wandb record one epoch
            wandbRecoder.record_epoch(epochID + 1, lossMean, accMean, aucMean)


    @staticmethod
    def epoch_train(model, dataLoader, optimizer, loss, scaler, cfg):
        model.cuda()
        model.train()

        lossHist = []
        gcl = 0.0
        dtime = 0

        progressBar = tqdm.tqdm(dataLoader, colour='white', file=sys.stdout)
        for batchID, sample in enumerate(progressBar):
            try:
                progressBar.write(' Epoch: {} | Global loss: {} | Running loss: {:1.5f} | Time: {}h {}m {}s '.format(
                        CheXExpertNet.currEpoch, gcl, np.mean(lossHist), int(dtime / 3600), int((dtime / 60) % 60),
                        dtime % 60)) if dtime > 0 else None

                target, image, segment = sample["lab"].cuda(), sample["img"].cuda(), sample["seg"].cuda()

                with autocast():
                    pred, logitMap = model(image, segment)
                    lossValue, _ = loss(pred=pred, gt=target, heatmaps=logitMap, seg=segment)

                optimizer.zero_grad()
                lossSum = lossValue[0] + lossValue[1] * cfg.train.scl_alpha
                scaler.scale(lossSum).backward()
                scaler.step(optimizer)
                scaler.update()

                gcl = lossValue.detach().cpu().numpy()
                gcl[1] = gcl[1] * cfg.train.scl_alpha
                lossHist.append(gcl)
                dtime = int(time.time() - CheXExpertNet.startTime)

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print(exception)
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise exception

    @staticmethod
    def epoch_val(model, dataLoader, loss, cfg):
        model.cuda()
        model.eval()

        with torch.no_grad():
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
            outLossItem = np.zeros(cfg.dataset.num_classes)

            progressBar = tqdm.tqdm(dataLoader, colour='green', file=sys.stdout)
            for batchID, sample in enumerate(progressBar):

                target, image = sample["lab"].cuda(), sample["img"].cuda()

                pred, logitMap = model(image)

                _, lossIdv = loss(pred=pred, gt=target, heatmaps=logitMap, seg=None)

                outLossItem += lossIdv[0]
                outGT = torch.cat((outGT, target), 0)
                # outPRED = torch.cat((outPRED, pred), 0)
                outPRED = torch.cat((outPRED, torch.cat(pred, 1)), 0)

                gc.collect()
                torch.cuda.empty_cache()

            lossIndividual = outLossItem / len(dataLoader)
            lossMean = lossIndividual.mean()

            aucIndividual, accIndividual, bestThreshholds = \
                computeAUCROC(outGT, outPRED, cfg.dataset.num_classes)

            aucMean = np.array(aucIndividual).mean()
            accMean = np.array(accIndividual).mean()

        return lossIndividual.tolist(), lossMean, aucIndividual, aucMean, accIndividual, accMean, bestThreshholds


    @staticmethod
    def test(cfg):
        # set mode
        cfg.mode = "test"

        # network architecture
        model = ChexClassifier(cfg).cuda()

        # data transforms
        transformSequence = DataTransforms(cfg).compile("test")

        # dataset builders
        datasetTest = DatasetGenerator(pathImageDirectory=cfg.dataset.images_path, pathDatasetFile=cfg.dataset.file_test,
                                      transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=cfg.val_test.batch_size, shuffle=False,
                                   num_workers=cfg.dataset.num_workers, pin_memory=True)

        # load checkpoint
        CheXExpertNet.load_checkpoint_from_cfg(cfg, model)

        # set CheXExpertNet.logger
        CheXExpertNet.logger = CheXExpertNet.build_logger_from_cfg(cfg)
        CheXExpertNet.logger.info(cfg)

        # start test
        model.eval()
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        with torch.no_grad():
            for i, sample in enumerate (tqdm.tqdm(dataLoaderTest, colour='green')):

                target, input = sample["lab"], sample["img"]

                varInput = torch.autograd.Variable(input).cuda()
                varTarget = torch.autograd.Variable(target).cuda()
                varOutput, logitMap = model(varInput)

                outGT = torch.cat((outGT, varTarget), 0)
                outPRED = torch.cat((outPRED, torch.cat(varOutput, 1)), 0)

                gc.collect()
                torch.cuda.empty_cache()

            aucIndividual, accIndividual, bestThreshholds = \
                computeAUCROC(outGT, outPRED, cfg.dataset.num_classes)

            aucMean = np.array(aucIndividual).mean()
            accMean = np.array(accIndividual).mean()

        summary = "meanAUC:{} AUC:{}\r\n meanACC:{} ACC:{}\r\n BestThresholds:{}"\
            .format(aucMean, aucIndividual, accMean, accIndividual, bestThreshholds)

        CheXExpertNet.logger.info(summary)

        return aucIndividual, aucMean


    @staticmethod
    def cam(cfg, showImg=False, iobbThreshold=None):
        import importlib
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # set mode
        cfg.mode = "test"

        # network architecture
        model = ChexClassifier(cfg).cuda()

        # load checkpoint
        CheXExpertNet.load_checkpoint_from_cfg(cfg, model)

        # data transforms
        transformSequence = DataTransforms(cfg).compile("test")

        # dataset builders
        datasetTest = BBoxDatasetGenerator(pathImageDirectory=cfg.dataset.images_path, pathDatasetFile=cfg.dataset.file_bbox,
                                      transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=cfg.val_test.batch_size, shuffle=False,
                                   num_workers=cfg.dataset.num_workers, collate_fn=BBoxCollater)

        # load pytorch_grad_cam package
        pytorchGradCamRoot = importlib.import_module("pytorch_grad_cam")

        targetLayer = [model.backbone.get_final_conv()]

        # Construct the CAM object once, and then re-use it on many images:
        cam = getattr(pytorchGradCamRoot, cfg.box_fusion.cam_list[0])(model=model, target_layers=targetLayer, use_cuda=True)

        tqdm.tqdm.write(f"\n Use {cfg.box_fusion.cam_list[0]} for positioning")

        listPLA = [np.array([])] * cfg.dataset.num_classes

        for sample in tqdm.tqdm(dataLoaderTest, colour='green'):
            oriInputBatch = sample["cop"]
            varInputBatch = torch.autograd.Variable(sample["img"]).cuda()
            for nItem in range(len(oriInputBatch)):
                oriInput = oriInputBatch[nItem]
                varInput = varInputBatch[nItem].unsqueeze_(0)
                count = 1
                for nClass in range(cfg.dataset.num_classes):
                    if nClass in sample["box"][nItem]:

                        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                        targets = [nClass] * cfg.dataset.num_classes
                        grayscale_cam = cam(input_tensor=varInput, targets=targets)

                        # In this example grayscale_cam has only one image in the batch:
                        grayscale_cam = grayscale_cam[0, :]
                        visualization = show_cam_on_image(oriInput/255.0, grayscale_cam)[:, :, ::-1]

                        # get pseudo bbox [x1,y1,x2,y2]
                        pseudoBox = []
                        _, bWeightMap = cv2.threshold(grayscale_cam, np.max(grayscale_cam) / 1.0001, 1.0, cv2.THRESH_BINARY)
                        contours, hierarchy = cv2.findContours((bWeightMap * 255).astype("uint8"), cv2.RETR_TREE,
                                                               cv2.CHAIN_APPROX_NONE)
                        for c in contours:
                            x, y, w, h = cv2.boundingRect(c)
                            pseudoBox.append([x, y, x + w, y + h])

                        pseudoBox = np.array(pseudoBox)

                        # get ground truth box [x1,y1,x2,y2]
                        gtBox = np.array([[sample["box"][nItem][nClass][0],
                                           sample["box"][nItem][nClass][2],
                                           sample["box"][nItem][nClass][1],
                                           sample["box"][nItem][nClass][3]]])

                        # generate heatmap
                        if showImg:
                            if count == 1:
                                plt.figure(figsize=(10, 5))
                                plt.subplot(2, 3, count)
                                plt.title(os.path.basename(sample["pth"][nItem]), pad=20)
                                plt.imshow(oriInput)
                            count += 1
                            plt.subplot(2, 3, count)
                            plt.title(cfg.dataset.class_names[nClass], pad=20)
                            plt.imshow(visualization)
                            ax = plt.gca()
                            ax.add_patch(plt.Rectangle(
                                (gtBox[0][0], gtBox[0][1]), gtBox[0][2] - gtBox[0][0], gtBox[0][3] - gtBox[0][1],
                                color="red", fill=False, linewidth=1))
                            for box in pseudoBox:
                                ax.add_patch(plt.Rectangle(
                                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                    color="black", fill=False, linewidth=1))
                            plt.show()
                        # calculate iobb
                        IoBB = computeIoBB(gtBox, pseudoBox)
                        listPLA[nClass] = np.append(listPLA[nClass], np.max(IoBB))

                # plt.savefig("./heatmaps/" + os.path.basename(sample["pth"][nItem]))
        print("iobb:")
        meanIoBB = 0
        meanIoBBList = []
        for n in range(8):
            mean = np.mean(listPLA[n])
            print(cfg.dataset.class_names[n] + ": " + str(mean))
            meanIoBB = meanIoBB + mean
            meanIoBBList.append(mean)
        print("mean iobb: ", meanIoBB/8)

        return meanIoBBList, meanIoBB


    @staticmethod
    def mult_cam(cfg, showImg=False):
        import importlib
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # set mode
        cfg.mode = "test"

        # network architecture
        models = []
        for item in cfg.val_test.model_path:
            models.append(ChexClassifier(cfg).cuda())

        # load checkpoint
        for idx, item in enumerate(cfg.val_test.model_path):
            CheXExpertNet.load_checkpoint_from_cfg(cfg, models[idx], idx)

        # data transforms
        transformSequence = DataTransforms(cfg).compile("test")

        # dataset builders
        datasetTest = BBoxDatasetGenerator(pathImageDirectory=cfg.dataset.images_path, pathDatasetFile=cfg.dataset.file_bbox,
                                      transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=cfg.val_test.batch_size, shuffle=False,
                                   num_workers=cfg.dataset.num_workers, collate_fn=BBoxCollater)

        # load pytorch_grad_cam package
        pytorchGradCamRoot = importlib.import_module("pytorch_grad_cam")

        # record file path
        dictImg = []

        # record ground truth boxes
        dictGtBoxes = [[None for i in range(cfg.dataset.num_classes)] for i in range(len(datasetTest))]

        # record pseudo boxes
        dictPseudoBoxesForModel = []

        # obtain CAM for every model
        for idx, model in enumerate(models):

            # set the target layer for CAM usage
            targetLayer = [model.backbone.get_final_conv()]

            # record pseudo boxes for each CAM
            dictPseudoBoxes = {}

            # load the CAM modules registered in cfg.box_fusion.cam_list
            cam_mod_list = [getattr(pytorchGradCamRoot, name) for name in cfg.box_fusion.cam_list]

            # obtain CAM for every image
            for cnt, CAM in enumerate(cam_mod_list):

                tqdm.tqdm.write(f"\n {cnt + 1 + idx * len(cam_mod_list)} / {len(cam_mod_list) * len(models)}"
                                f" Model {cfg.val_test.model_path[idx]} use {CAM.__name__} for positioning")

                # Construct the CAM object once, and then re-use it on many images:
                cam = CAM(model=model, target_layers=targetLayer, use_cuda=True)

                pseudoBoxes = [[np.zeros(shape=(1, 4)) for i in range(cfg.dataset.num_classes)] for i in range(len(datasetTest))]

                itemCount = 0
                for sample in tqdm.tqdm(dataLoaderTest, colour='green'):
                    oriInputBatch = sample["cop"]
                    varInputBatch = torch.autograd.Variable(sample["img"]).cuda()
                    for nItem in range(len(oriInputBatch)):
                        oriInput = oriInputBatch[nItem]

                        if len(dictImg) < len(dataLoaderTest.dataset):
                            dictImg.append(oriInput)

                        varInput = varInputBatch[nItem].unsqueeze_(0)
                        subImgCnt = 0
                        for nClass in range(cfg.dataset.num_classes):
                            if nClass in sample["box"][nItem]:

                                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                                targets = [nClass] * cfg.dataset.num_classes
                                grayscale_cam = cam(input_tensor=varInput, targets=targets)

                                # In this example grayscale_cam has only one image in the batch:
                                grayscale_cam = grayscale_cam[0, :]

                                # get pseudo bbox [x1,y1,x2,y2]
                                pseudoBox = []
                                _, bWeightMap = cv2.threshold(grayscale_cam, np.max(grayscale_cam) / 2.0, 1.0, cv2.THRESH_BINARY)
                                contours, hierarchy = cv2.findContours((bWeightMap * 255).astype("uint8"), cv2.RETR_TREE,
                                                                       cv2.CHAIN_APPROX_NONE)
                                for c in contours:
                                    x, y, w, h = cv2.boundingRect(c)
                                    pseudoBox.append([x, y, x + w, y + h])

                                pseudoBox = np.array(pseudoBox)

                                # get ground truth box [x1,y1,x2,y2]
                                gtBox = np.array([[sample["box"][nItem][nClass][0],
                                                   sample["box"][nItem][nClass][2],
                                                   sample["box"][nItem][nClass][1],
                                                   sample["box"][nItem][nClass][3]]])

                                pseudoBoxes[itemCount][nClass] = pseudoBox
                                dictGtBoxes[itemCount][nClass] = gtBox

                                # generate heatmap
                                visualization = show_cam_on_image(oriInput / 255.0, grayscale_cam)[:, :, ::-1]
                                if subImgCnt == 0:
                                    plt.figure(figsize=(10, 5))
                                    plt.subplot(2, 3, 1)
                                    plt.title(os.path.basename(sample["pth"][nItem]), pad=20)
                                    plt.imshow(oriInput)
                                    subImgCnt += 1
                                plt.subplot(2, 3, subImgCnt + 1)
                                plt.title(cfg.dataset.class_names[nClass], pad=20)

                                plt.imshow(visualization)
                                ax = plt.gca()
                                ax.add_patch(plt.Rectangle(
                                    (gtBox[0][0], gtBox[0][1]), gtBox[0][2] - gtBox[0][0], gtBox[0][3] - gtBox[0][1],
                                    color="red", fill=False, linewidth=1))
                                for box in pseudoBox:
                                    ax.add_patch(plt.Rectangle(
                                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                        color="black", fill=False, linewidth=1))
                                subImgCnt += 1

                        itemCount += 1
                        item_name = os.path.basename(sample["pth"][nItem])

                        if showImg:
                            plt.show()
                        plt.savefig(f"./heatmaps/{CAM.__name__}/{idx}_{item_name}")
                        plt.close()

                dictPseudoBoxes[CAM.__name__] = pseudoBoxes
            dictPseudoBoxesForModel.append(dictPseudoBoxes)

    # CheXExpertNet.pso(cfg, dictPseudoBoxesForModel, dictGtBoxes)

        return dictPseudoBoxesForModel, dictGtBoxes, dictImg


    @staticmethod
    def iobb(cfg, dictPseudoBoxesForModel, dictGtBoxes, cam_weight):

        totalIobb = 0
        meanIobbForModel = []
        meanIobbListForModel = []

        for modelIdx, dictPseudoBoxes in enumerate(dictPseudoBoxesForModel):
            # Weighted boxes fusion
            listWeightedBoxes = [[np.zeros(shape=(1, 4)) for i in range(cfg.dataset.num_classes_with_boxes)] for i in range(len(dictGtBoxes))]

            for item in range(len(dictGtBoxes)):
                for nClass in range(cfg.dataset.num_classes_with_boxes):
                    if dictGtBoxes[item][nClass] is not None:
                        weightedBoxes = np.zeros(shape=(1, 4))
                        sum = 0
                        for camIdx, cam_name in enumerate(cfg.box_fusion.cam_list):
                            weight = cam_weight[modelIdx][camIdx][nClass]
                            weightedBoxes += dictPseudoBoxes[cam_name][item][nClass] * weight
                            sum += weight
                        weightedBoxes /= sum
                        listWeightedBoxes[item][nClass] = weightedBoxes

            # calculate iobb
            listPLA = [np.array([])] * cfg.dataset.num_classes_with_boxes
            for item in range(len(dictGtBoxes)):
                for nClass in range(cfg.dataset.num_classes_with_boxes):
                    gtBox = dictGtBoxes[item][nClass]
                    WeightedBox = listWeightedBoxes[item][nClass]
                    if gtBox is not None and WeightedBox is not None:
                        iobb = computeIoBB(gtBox, WeightedBox)
                        listPLA[nClass] = np.append(listPLA[nClass], np.max(iobb))

            print("iobb:")
            meanIobb = 0
            meanIobbList = []
            for n in range(cfg.dataset.num_classes_with_boxes):
                mean = np.mean(listPLA[n])
                print(cfg.dataset.class_names[n] + ": " + str(mean))
                meanIobb = meanIobb + mean
                meanIobbList.append(mean)
            print("meanIobb: ", meanIobb/cfg.dataset.num_classes_with_boxes)

            totalIobb += meanIobb
            meanIobbForModel.append(meanIobb)
            meanIobbListForModel.append(meanIobbList)

        return totalIobb, meanIobbForModel, meanIobbListForModel

    @staticmethod
    def region_fuser(cfg, dictPseudoBoxesForModel, dictGtBoxes, dictImg, showImg=True):
        fuser = RegionFuser(cfg)

        # set logger
        CheXExpertNet.logger = CheXExpertNet.build_logger_from_cfg(cfg)

        # optimization
        mean_iobb, weight_list = fuser.optimize(dictPseudoBoxesForModel, dictGtBoxes)

        # T(IOR)
        thr_list = [0.10, 0.25, 0.50, 0.75, 0.90]
        for thr in thr_list:
            meanIobb, meanIobbList, listIobb, listWeightedBoxes = fuser.eval(dictPseudoBoxesForModel, dictGtBoxes, weight_list, thr)
            print("\nThreshold ", thr)
            for n in range(cfg.dataset.num_classes_with_boxes):
                CheXExpertNet.logger.info(cfg.dataset.class_names[n] + ": " + str(meanIobbList[n]))
                print(cfg.dataset.class_names[n] + ": " + str(meanIobbList[n]))

        # show and save img with boxes
        for nItem, _ in enumerate(tqdm.tqdm(dictGtBoxes, colour='green')):
            count = 1
            img = dictImg[nItem]
            for nClass in range(cfg.dataset.num_classes_with_boxes):
                gtBox = dictGtBoxes[nItem][nClass]

                if gtBox is not None and not np.all(gtBox == 0):
                    pseudoBox = listWeightedBoxes[nItem][nClass]

                    # generate heatmap
                    if count == 1:
                        plt.figure(figsize=(10, 5))
                        plt.subplot(2, 3, count)
                        plt.title(nItem, pad=20)
                        plt.imshow(img)
                    plt.subplot(2, 3, count)
                    plt.title(cfg.dataset.class_names[nClass], pad=20)
                    plt.imshow(img)
                    ax = plt.gca()
                    ax.add_patch(plt.Rectangle(
                        (gtBox[0][0], gtBox[0][1]), gtBox[0][2]-gtBox[0][0], gtBox[0][3]-gtBox[0][1],
                        color="red", fill=False, linewidth=1))
                    for box in pseudoBox:
                        ax.add_patch(plt.Rectangle(
                            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            color="yellow", fill=False, linewidth=1))
                    count += 1
            # if showImg:
            #     plt.show()
            plt.savefig(f"./heatmaps/{nItem}")
            plt.close()

        return

    @staticmethod
    def class_fuser(cfg):
        # set mode
        cfg.mode = "test"

        # network architecture
        models = []
        for item in cfg.val_test.model_path:
            models.append(ChexClassifier(cfg).cuda())

        # load checkpoint
        for idx, item in enumerate(cfg.val_test.model_path):
            CheXExpertNet.load_checkpoint_from_cfg(cfg, models[idx], idx)

        # set CheXExpertNet.logger
        CheXExpertNet.logger = CheXExpertNet.build_logger_from_cfg(cfg)
        CheXExpertNet.logger.info(cfg)

        # data transforms
        transformSequence = DataTransforms(cfg).compile("test")

        # dataset builders
        datasetTest = DatasetGenerator(pathImageDirectory=cfg.dataset.images_path, pathDatasetFile=cfg.dataset.file_test,
                                      transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=cfg.val_test.batch_size, shuffle=False,
                                   num_workers=cfg.dataset.num_workers, pin_memory=True)

        # record ground truth boxes
        listGtLabel = []
        listProbForModel = []
        listFusedLabel = []

        # obtain CAM for every model
        for mIdx, model in enumerate(models):
            # start test
            model.eval()
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()

            with torch.no_grad():
                for i, sample in enumerate(tqdm.tqdm(dataLoaderTest, colour='green')):
                    target, input = sample["lab"], sample["img"]

                    varInput = torch.autograd.Variable(input).cuda()
                    varTarget = torch.autograd.Variable(target).cuda()
                    varOutput, logitMap = model(varInput)

                    outGT = torch.cat((outGT, varTarget), 0)
                    outPRED = torch.cat((outPRED, torch.cat(varOutput, 1)), 0)

                    gc.collect()
                    torch.cuda.empty_cache()

                listProbForModel.append(torch.sigmoid(outPRED).cpu().numpy())
                # listProbForModel.append(outPRED)
                listGtLabel = outGT.cpu().numpy()

        # Fuse outPRED
        for iidx in range(len(listProbForModel[0])):
            candidates = np.array([listProbForModel[midx][iidx] for midx in range(len(listProbForModel))])
            listFusedLabel.append([np.maximum(candidates[cnt], candidates[cnt + 1]) for cnt in range(0, len(listProbForModel)-1)])

        listFusedLabel = np.squeeze(np.array(listFusedLabel))

        aucIndividual, accIndividual, bestThreshholds = \
            computeAUCROC(listGtLabel, listFusedLabel, cfg.dataset.num_classes)

        aucMean = np.array(aucIndividual).mean()
        accMean = np.array(accIndividual).mean()

        summary = "meanAUC:{} AUC:{}\r\n meanACC:{} ACC:{}\r\n BestThresholds:{}"\
            .format(aucMean, aucIndividual, accMean, accIndividual, bestThreshholds)

        CheXExpertNet.logger.info(summary)

        return aucIndividual, aucMean


    @staticmethod
    def build_optimizer_from_cfg(params, cfg):
        if cfg.train.optimizer == 'SGD':
            return SGD(params, lr=cfg.train.lr, momentum=cfg.train.momentum,
                       weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'Adadelta':
            return Adadelta(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'Adagrad':
            return Adagrad(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'Adam':
            return Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'RMSprop':
            return RMSprop(params, lr=cfg.train.lr, momentum=cfg.train.momentum,
                           weight_decay=cfg.train.weight_decay)
        else:
            raise Exception('Unknown optimizer : {}'.format(cfg.train.optimizer))


    @staticmethod
    def build_loss_from_cfg(cfg):
        losses = Losses(cfg)
        losses.compile()
        return losses


    @staticmethod
    def build_logger_from_cfg(cfg):
        log_name = time.strftime("_%m-%d_%Hh%Mm%Ss", time.localtime())
        log_name = cfg.mode + log_name + ".txt"
        logger = build_logger(log_name, cfg.logger.log_path+"heatmap_size_"+str(cfg.model.heatmap_size), cfg.logger.enable, when=cfg.logger.Rotating)

        return logger


    @staticmethod
    def load_checkpoint_from_cfg(cfg, model, modelIdx=0, optimizer=None):
        epochStart = 0
        if cfg.mode == "train":
            model_path = cfg.model.check_point
        else:
            model_path = cfg.val_test.model_path

        if model_path:

            if isinstance(model_path, list):
                model_path = model_path[modelIdx]

            modelCheckpoint = torch.load(model_path)
            model.load_state_dict(modelCheckpoint['state_dict'])
            if cfg.mode == "train":
                optimizer.load_state_dict(modelCheckpoint['optimizer'])
                epochStart = modelCheckpoint['epoch']
            print(f"\nload checkpoint from {model_path} epoch {epochStart}\n")

        return epochStart
