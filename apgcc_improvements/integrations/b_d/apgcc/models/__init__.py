from .APGCC import Model_builder, SetCriterion_Crowd
from .matcher import build_matcher_crowd

# create the main model
def build_model(cfg, training):
   model = Model_builder(cfg)
   if not training: 
      return model

   weight_dict = dict(cfg.MODEL.WEIGHT_DICT)
   matcher = build_matcher_crowd(cfg)
   if not cfg.MODEL.AUX_EN:
      weight_dict.pop('loss_aux', None)
   if not cfg.MODEL.get('DENSE_AUX_EN', False):
      weight_dict.pop('loss_dense', None)
   criterion = SetCriterion_Crowd(num_classes=1, \
                                  matcher=matcher, weight_dict=weight_dict, \
                                  eos_coef=cfg.MODEL.EOS_COEF, \
                                  aux_kwargs = {'AUX_NUMBER': cfg.MODEL.AUX_NUMBER,
                                                'AUX_RANGE': cfg.MODEL.AUX_RANGE,
                                                'AUX_kwargs': cfg.MODEL.AUX_kwargs},
                                  edge_band=cfg.MODEL.get('EDGE_BAND', 0),
                                  edge_weight=cfg.MODEL.get('EDGE_WEIGHT', 0.0),
                                  dense_aux_en=cfg.MODEL.get('DENSE_AUX_EN', False),
                                  dense_sigma=cfg.MODEL.get('DENSE_SIGMA', 2.0))
   return model, criterion
