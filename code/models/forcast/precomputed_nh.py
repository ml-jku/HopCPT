import logging
from pathlib import Path
import pickle
from typing import Optional, Tuple
import numpy as np
import torch

from models.forcast.forcast_base import ForcastMode, ForcastModel, PredictionOutputType, FCModelPrediction, \
    FCPredictionData, FcSingleModelPrediction

LOGGER = logging.getLogger(__name__)


class PrecomputedNeuralHydrologyForcast(ForcastModel):
    """Model which returns precomputed predictions from a NeuralHydrology model.
    
    To train the neuralhydrology model, run: `nh-run train configuration/hydrology/nh-config.yaml`.
    To pre-calculate the predictions, run: `nh-run evaluate --run-dir outputs/neuralhydrology/regression_camels_XYZ`.
    """
    def __init__(self, **kwargs):
        super().__init__(forcast_mode=ForcastMode.ALL_ON_TRAIN, supported_outputs=(PredictionOutputType.POINT,))
        self._neuralhydrology_results_path = Path(kwargs['model_params']['nh_results_path'])
        self._target_variable = kwargs['model_params']['target_variable']
        assert self._neuralhydrology_results_path.is_file()

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        with self._neuralhydrology_results_path.open('rb') as f:
            precomputed_predictions = pickle.load(f)
        LOGGER.info(f'Loaded NeuralHydrology predictions from {self._neuralhydrology_results_path}.')

        if precalc_fc_steps is not None:
            basin = kwargs['ts_id'].split('_')[0]
            slice_end = kwargs['X_full'].shape[0] - Y.shape[0]
            prediction = self._slice_predictions(precomputed_predictions[basin], 0, slice_end)

            y_mean, y_std = kwargs['y_normalize_props']
            prediction = (torch.from_numpy(prediction) - y_mean) / y_std
            return FcSingleModelPrediction(point=prediction[..., None]), None
        return None

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        raise NotImplemented()  # Possible but not needed.

    def _check_pred_data(self, pred_data: FCPredictionData):
        # We don't actually use these since everything is precomputed, but in theory (if we'd re-run the model), we
        # would need this information.
        assert pred_data.X_step is not None
        assert pred_data.X_past is not None

    def can_handle_different_alpha(self):
        return False

    @property
    def train_per_time_series(self):
        return True

    @property
    def uses_past_for_prediction(self):
        return True

    def _slice_predictions(self, basin_predictions, slice_start: int, slice_end: int) -> np.ndarray:
        precomputed_basin = basin_predictions['1D']['xr'][self._target_variable].sel(time_step=0)
        Y_hat = precomputed_basin.isel(date=slice(slice_start, slice_end)).to_numpy()
        assert np.isnan(Y_hat).sum() == 0
        assert len(Y_hat) == len(precomputed_basin)  # Check to make sure we didn't mess up train/val splits.
        return Y_hat
