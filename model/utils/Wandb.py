import time
import wandb

class WandbRecoder(object):

    def __init__(self, cfg, model=None):

        self._cfg = cfg
        if cfg.wandb.enable:
            timestamp = time.strftime("_%m-%d_%Hh%Mm%Ss", time.localtime())
            w = wandb.init(project='ChexNet++',
                           name=timestamp,
                           config=cfg,
                           resume="allow")
            self._wandb = w

            if model:
                self._wandb.watch(model)

        else:
            self._wandb = None


    def record_epoch(self, epochID, lossMean, accMean, aucMean):
        if self._wandb:
            self._wandb.log({"epoch": epochID,
                             "lossMean": lossMean,
                             "accMean": accMean,
                             "aucMean": aucMean,
                             })

    def record_iter(self, iter, lossMean, lossVal):
        self._wandb.log({"iter": iter,
                         "iterLossMean": lossMean,
                         "iterLossVal": lossVal
                         })
