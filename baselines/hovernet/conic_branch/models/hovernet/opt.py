import torch.optim as optim
from run_utils.callbacks.base import (AccumulateRawOutput, PeriodicSaver,
                                      ProcessAccumulatedRawOutput,
                                      ScalarMovingAverage, ScheduleLr, TrackLr,
                                      TriggerEngine, VisualizeOutput)
from run_utils.callbacks.logging import LoggingEpochOutput
from run_utils.engine import Events

from .net_desc import create_model
from .run_desc import (proc_valid_step_output, train_step, valid_step,
                       viz_step_output)


def get_config(
        train_loader_list,
        infer_loader_list,
        loader_kwargs={},
        model_kwargs={},
        optimizer_kwargs={},
        **kwargs):

    config = {
        # Phase 0: freeze encoder, 训练 decoder 分支 (50 epoch)
        # Phase 1: 全模型微调 (40 epoch, 无早停, 训练完成后手动选最佳)
        'phase_list': [
            {
                'run_info': {
                    'net': {
                        'desc': lambda: create_model(
                            freeze=True, **model_kwargs),
                        'optimizer': [
                            optim.Adam,
                            {
                                'lr': 1.0e-4,
                                'betas': (0.9, 0.999),
                            },
                        ],
                        'lr_scheduler': (
                            lambda opt, n_iter:
                                optim.lr_scheduler.StepLR(opt, 25)),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        'pretrained': None,
                    },
                },
                'target_info': {
                    'gen': (None, {}),
                    'viz': (None, {})
                },
                'loader': loader_kwargs,
                'nr_epochs': 35,
            },
            {
                'run_info': {
                    'net': {
                        'desc': lambda: create_model(
                            freeze=False, **model_kwargs),
                        'optimizer': [
                            optim.Adam,
                            {
                                'lr': 1.0e-4,
                                'betas': (0.9, 0.999),
                            },
                        ],
                        'lr_scheduler': (
                            lambda opt, n_iter:
                                optim.lr_scheduler.StepLR(opt, 25)),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        'pretrained': -1,
                    },
                },
                'target_info': {
                    'gen': (None, {}),
                    'viz': (None, {})
                },
                'loader': loader_kwargs,
                'nr_epochs': 40,
            },
        ],

        'run_engine': {
            'train': {
                'loader': train_loader_list,
                'run_step': train_step,
                'reset_per_run': False,

                'callbacks': {
                    Events.STEP_COMPLETED: [
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        VisualizeOutput(viz_step_output),
                        LoggingEpochOutput(),
                        TriggerEngine("infer"),
                        ScheduleLr(),
                    ],
                },
            },
            'infer': {
                'loader': infer_loader_list,
                'run_step': valid_step,
                'reset_per_run': True,

                'callbacks': {
                    Events.STEP_COMPLETED: [
                        AccumulateRawOutput()
                    ],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedRawOutput(
                            lambda name, data: proc_valid_step_output(
                                data, num_types=model_kwargs['num_types'])
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }

    return config
