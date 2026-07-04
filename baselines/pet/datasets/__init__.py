import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .CellPoints import build as build_cell_points

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'CELL': './data/CellPoints',
    'BCData': './data/BCData_pet',
    'CoNIC': './data/CoNIC_pet',
    'MoNuSeg': './data/MoNuSeg_pet',
}

def build_dataset(image_set, args):
    default_data_path = './data/ShanghaiTech/PartA'
    if args.data_path == default_data_path and args.dataset_file in data_path:
        args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file in {'CELL', 'BCData', 'CoNIC', 'MoNuSeg'}:
        return build_cell_points(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
