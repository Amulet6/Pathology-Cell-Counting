from .build import loading_data

def build_dataset(cfg, eval_list=None):
    return loading_data(cfg, eval_list=eval_list)