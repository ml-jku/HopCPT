from typing import Tuple, Optional, List

import numpy as np

from models.forcast.forcast_base import FCPredictionData, PredictionOutputType
from models.uncertainty.pi_base import PIModel, PICalibData, PIModelPrediction, PIPredictionStepData, \
     PICalibArtifacts
from utils.calc_np import calc_residuals, calc_default_conformal_Q


class DefaultConformal(PIModel):
    """
    Simple model which uses the default conformal approach and considers the quantiles of the calibration data
    """
    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,),
                         ts_ids=kwargs["ts_ids"])
        self.eps_calib = None

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
                             X_step=calib_data.X_calib, step_offset=calib_data.step_offset), retrieve_tensor=False).point
        self.eps_calib = calc_residuals(y_hat=Y_hat, y=calib_data.Y_calib.numpy())
        return PICalibArtifacts(fc_Y_hat=Y_hat, eps=self.eps_calib)

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        alpha, X_step, X_past, Y_past = pred_data.alpha, pred_data.X_step, pred_data.X_past, pred_data.Y_past
        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=pred_data.step_offset_overall)).point

        #Default Style
        width = calc_default_conformal_Q(self.eps_calib, alpha)
        pred_int = Y_hat - width, Y_hat + width
        #EnPI Style (without beta)
        #q_conformal_low = np.quantile(self.eps_calib, alpha / 2)
        #q_conformal_high = np.quantile(self.eps_calib, (1 - alpha / 2))
        #pred_int = Y_hat + q_conformal_low, Y_hat + q_conformal_high
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    def model_ready(self):
        return self.eps_calib is not None

    @property
    def can_handle_different_alpha(self):
        return True


class DefaultConformalPlusRecent(DefaultConformal):
    """
    Simple model which uses the default conformal approach and considers the quantiles of the calibration data
    + k (window_lenght) recent residuals
    """
    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,),
                         ts_ids=kwargs["ts_ids"])
        self._past_window_len = kwargs['past_window_len']

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        alpha, X_step, X_past, Y_past, eps_past = pred_data.alpha, pred_data.X_step, pred_data.X_past, \
                                                  pred_data.Y_past, pred_data.eps_past
        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=pred_data.step_offset_overall)).point

        #Default Style
        width = calc_default_conformal_Q(np.concatenate((self.eps_calib, eps_past.numpy())), alpha)
        pred_int = Y_hat - width, Y_hat + width
        #EnPI Style (without beta)
        #q_conformal_low = np.quantile(np.concatenate((self.eps_calib, eps_past)), alpha / 2)
        #q_conformal_high = np.quantile(np.concatenate((self.eps_calib, eps_past)), (1 - alpha / 2))
        #pred_int = Y_hat + q_conformal_low, Y_hat + q_conformal_high
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    def required_past_len(self) -> Tuple[int, int]:
        fc_required_len = super().required_past_len()
        return max(fc_required_len[0], self._past_window_len), max(fc_required_len[1], self._past_window_len)

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.eps_past is not None