# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
print(torchvision.__version__)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
    
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

# For Localizaiton
from scipy import spatial as ss
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])
    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]
    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False
    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum): vis[i] = False
        if dfs(a): ans += 1
    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True
    return ans, assign

def compute_tp(pred,gt, threshold):
    if len(pred) == 0 or gt.size(0) == 0:
        return 0
    dist_matrix = ss.distance_matrix(pred,gt,p=2)
    match_matrix = np.zeros(dist_matrix.shape,dtype=bool)
    for i_pred_p in range(len(pred)):
        pred_dist = dist_matrix[i_pred_p,:]    
        match_matrix[i_pred_p,:] = pred_dist<=threshold

    tp, assign = hungarian(match_matrix)
    
    return tp


def _points_to_numpy(points):
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float32)


def _greedy_radius_filter(points, scores, radius, limit=None):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    order = np.argsort(-scores)
    kept_pts = []
    kept_scores = []
    r2 = float(radius) * float(radius)
    for idx in order:
        pt = points[idx]
        if kept_pts:
            prev = np.asarray(kept_pts, dtype=np.float32)
            if np.min(np.sum((prev - pt[None, :]) ** 2, axis=1)) < r2:
                continue
        kept_pts.append(pt.astype(np.float32))
        kept_scores.append(float(scores[idx]))
        if limit is not None and len(kept_pts) >= limit:
            break
    return np.asarray(kept_pts, dtype=np.float32), np.asarray(kept_scores, dtype=np.float32)


def _sample_dense_scores(points, dense_prob, img_shape):
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)

    img_h, img_w = img_shape
    h, w = dense_prob.shape[-2:]
    coords = torch.as_tensor(points, dtype=torch.float32, device=dense_prob.device)
    xs = coords[:, 0] / max(float(img_w - 1), 1.0) * 2.0 - 1.0
    ys = coords[:, 1] / max(float(img_h - 1), 1.0) * 2.0 - 1.0
    grid = torch.stack([xs, ys], dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        dense_prob,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return sampled.view(-1).detach().cpu().numpy().astype(np.float32)


def extract_dense_points(outputs, sample_idx, score_threshold=0.35, kernel=5, topk=256):
    dense_map = outputs.get('dense_map')
    if dense_map is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    dense_prob = torch.sigmoid(dense_map[sample_idx:sample_idx + 1])
    pad = max(int(kernel) // 2, 0)
    pooled = F.max_pool2d(dense_prob, kernel_size=kernel, stride=1, padding=pad)
    keep = (dense_prob >= score_threshold) & (dense_prob == pooled)
    ys, xs = torch.where(keep[0, 0])
    if ys.numel() == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scores = dense_prob[0, 0, ys, xs]
    if topk is not None and ys.numel() > topk:
        top_scores, order = torch.topk(scores, k=int(topk))
        ys = ys[order]
        xs = xs[order]
        scores = top_scores

    img_h, img_w = outputs['img_shape']
    h, w = dense_prob.shape[-2:]
    xs = xs.float() / max(float(w - 1), 1.0) * max(float(img_w - 1), 1.0)
    ys = ys.float() / max(float(h - 1), 1.0) * max(float(img_h - 1), 1.0)
    points = torch.stack([xs, ys], dim=1).detach().cpu().numpy().astype(np.float32)
    scores = scores.detach().cpu().numpy().astype(np.float32)
    return points, scores


def _merge_points_with_radius(base_points, add_points, radius, limit=None):
    fused = [pt.tolist() if isinstance(pt, np.ndarray) else list(pt) for pt in base_points]
    if len(fused) > 0:
        fused_arr = np.asarray(fused, dtype=np.float32)
    else:
        fused_arr = np.empty((0, 2), dtype=np.float32)
    r2 = float(radius) * float(radius)
    added = 0
    for pt in add_points:
        pt_arr = np.asarray(pt, dtype=np.float32)
        if len(fused_arr) > 0:
            if np.min(np.sum((fused_arr - pt_arr[None, :]) ** 2, axis=1)) < r2:
                continue
        fused.append(pt_arr.tolist())
        fused_arr = np.asarray(fused, dtype=np.float32)
        added += 1
        if limit is not None and added >= limit:
            break
    return fused


def get_fused_predictions(outputs, cfg_or_threshold=0.5, sample_idx=0):
    if hasattr(cfg_or_threshold, 'MODEL'):
        cfg = cfg_or_threshold
        point_threshold = float(cfg.TEST.THRESHOLD)
        fusion_en = bool(getattr(cfg.MODEL, 'DENSE_FUSION_EN', False))
        dense_threshold = float(getattr(cfg.MODEL, 'DENSE_FUSION_THRESHOLD', 0.35))
        dense_radius = float(getattr(cfg.MODEL, 'DENSE_FUSION_RADIUS', 6.0))
        dense_kernel = int(getattr(cfg.MODEL, 'DENSE_FUSION_KERNEL', 5))
        dense_topk = int(getattr(cfg.MODEL, 'DENSE_FUSION_TOPK', 256))
        dense_max_add = int(getattr(cfg.MODEL, 'DENSE_FUSION_MAX_ADD', 128))
        low_conf_threshold = float(getattr(cfg.MODEL, 'DENSE_FUSION_LOW_CONF', 0.20))
        candidate_dense_threshold = float(getattr(cfg.MODEL, 'DENSE_FUSION_CANDIDATE_DENSE', 0.55))
        candidate_topk = int(getattr(cfg.MODEL, 'DENSE_FUSION_CANDIDATE_TOPK', 64))
        raw_max_add = int(getattr(cfg.MODEL, 'DENSE_FUSION_RAW_MAX_ADD', 0))
    else:
        cfg = None
        point_threshold = float(cfg_or_threshold)
        fusion_en = False
        dense_threshold = 0.35
        dense_radius = 6.0
        dense_kernel = 5
        dense_topk = 256
        dense_max_add = 128
        low_conf_threshold = min(point_threshold, 0.20)
        candidate_dense_threshold = 0.55
        candidate_topk = 64
        raw_max_add = 0

    point_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][sample_idx]
    point_xy = outputs['pred_points'][sample_idx]
    keep = point_scores > point_threshold
    main_points = point_xy[keep].detach().cpu().numpy().astype(np.float32)
    main_scores = point_scores[keep].detach().cpu().numpy().astype(np.float32)

    if not fusion_en or 'dense_map' not in outputs:
        return main_points.tolist(), len(main_points)

    dense_prob = torch.sigmoid(outputs['dense_map'][sample_idx:sample_idx + 1])
    img_shape = outputs['img_shape']
    fused = main_points.tolist()

    recover_mask = (point_scores > low_conf_threshold) & (~keep)
    recover_points = point_xy[recover_mask].detach().cpu().numpy().astype(np.float32)
    recover_scores = point_scores[recover_mask].detach().cpu().numpy().astype(np.float32)
    if len(recover_points) > 0:
        dense_support = _sample_dense_scores(recover_points, dense_prob, img_shape)
        valid = dense_support >= candidate_dense_threshold
        if np.any(valid):
            recover_points = recover_points[valid]
            recover_scores = recover_scores[valid] * dense_support[valid]
            recover_points, _ = _greedy_radius_filter(
                recover_points,
                recover_scores,
                dense_radius,
                limit=min(candidate_topk, dense_max_add),
            )
            fused = _merge_points_with_radius(main_points, recover_points, dense_radius, limit=dense_max_add)

    remain_add = max(dense_max_add - max(len(fused) - len(main_points), 0), 0)
    if raw_max_add > 0 and remain_add > 0:
        dense_points, dense_scores = extract_dense_points(
            outputs, sample_idx,
            score_threshold=dense_threshold,
            kernel=dense_kernel,
            topk=dense_topk,
        )
        if len(dense_points) > 0:
            dense_points, dense_scores = _greedy_radius_filter(
                dense_points,
                dense_scores,
                dense_radius,
                limit=min(raw_max_add, remain_add),
            )
            fused = _merge_points_with_radius(
                np.asarray(fused, dtype=np.float32),
                dense_points,
                dense_radius,
                limit=min(raw_max_add, remain_add),
            )
    return fused, len(fused)

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
