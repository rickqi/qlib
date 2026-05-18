# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging

_logger = logging.getLogger(__name__)

try:
    from .catboost_model import CatBoostModel
except ModuleNotFoundError:
    CatBoostModel = None
    _logger.debug("CatBoostModel skipped (install catboost to enable)")
try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    _logger.debug("DEnsembleModel and LGBModel skipped (install lightgbm to enable)")
try:
    from .xgboost import XGBModel
except ModuleNotFoundError:
    XGBModel = None
    _logger.debug("XGBModel skipped (install xgboost to enable)")
try:
    from .linear import LinearModel
except ModuleNotFoundError:
    LinearModel = None
    _logger.debug("LinearModel skipped (install scipy and sklearn to enable)")
# import pytorch models
try:
    from .pytorch_alstm import ALSTM
    from .pytorch_gats import GATs
    from .pytorch_gru import GRU
    from .pytorch_lstm import LSTM
    from .pytorch_nn import DNNModelPytorch
    from .pytorch_tabnet import TabnetModel
    from .pytorch_sfm import SFM_Model
    from .pytorch_tcn import TCN
    from .pytorch_add import ADD

    pytorch_classes = (ALSTM, GATs, GRU, LSTM, DNNModelPytorch, TabnetModel, SFM_Model, TCN, ADD)
except ModuleNotFoundError:
    pytorch_classes = ()
    _logger.debug("PyTorch models skipped (install torch to enable)")

all_model_classes = (CatBoostModel, DEnsembleModel, LGBModel, XGBModel, LinearModel) + pytorch_classes
