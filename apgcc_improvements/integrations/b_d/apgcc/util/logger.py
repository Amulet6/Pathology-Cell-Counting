import os, sys
import logging
import numpy as np
import cv2
import torchvision.transforms as standard_transforms

################################################
## Record Logger:
## 1. setup_logger, writer recorder.
## 2. AvgerageMeter, record Meter.
################################################
# Logger file
def setup_logger(name, save_dir, distributed_rank, train=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt" if train else 'log_eval.txt'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

## Model logger
class AvgerageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class EvaluateMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.MAE_avg = 0
        self.MAE_sum = 0
        self.MAE_min = 1e6
        self.MSE_avg = 0
        self.MSE_sum = 0
        self.MSE_min = 1e6
        self.cnt = 0
        self.best_ep = 0 

    def update(self, mae, mse, ep=0, n=1):
        self.MAE_sum += float(mae) * n
        self.MSE_sum += float(mse) * n
        self.cnt += n
        self.MAE_avg = self.MAE_sum / self.cnt
        self.MSE_avg = self.MSE_sum / self.cnt
        if mae < self.MAE_min :
            self.MAE_min = mae
            self.MSE_min = mse
            self.best_ep = ep

## Visualization
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    # samples -> tensor: [batch, 3, H, W]
    # targets -> list of dict: [{'points':[], 'image_id': str}]
    # pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]
    pil_to_tensor = standard_transforms.ToTensor()
    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)
        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        # combined image: GT (green) + Pred (red) on the same canvas
        sample_combined = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        for t in gts[idx]:
            sample_combined = cv2.circle(sample_combined, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        for p in pred[idx]:
            sample_combined = cv2.circle(sample_combined, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        prefix = '{}_{}_{}'.format(int(name), des, 'gt{}_pred{}'.format(len(gts[idx]), len(pred[idx]))) \
            if des is not None else '{}_gt{}_pred{}'.format(int(name), len(gts[idx]), len(pred[idx]))
        cv2.imwrite(os.path.join(vis_dir, prefix + '_gt.jpg'), sample_gt)
        cv2.imwrite(os.path.join(vis_dir, prefix + '_pred.jpg'), sample_pred)
        cv2.imwrite(os.path.join(vis_dir, prefix + '_combined.jpg'), sample_combined)