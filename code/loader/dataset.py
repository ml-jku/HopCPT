from abc import abstractmethod, ABC
from typing import Optional, Tuple

import torch


class TsDataset(ABC):
    def _normalize(self, data_full):
        mean = torch.mean(data_full[:self.test_step], dim=0)
        std = torch.std(data_full[:self.test_step], dim=0)
        return (data_full - mean) / std, mean, std

    @property
    @abstractmethod
    def X_normalize_props(self) -> Tuple[float, float]:
        pass

    @property
    @abstractmethod
    def Y_normalize_props(self) -> Tuple[float, float]:
        pass

    @property
    @abstractmethod
    def ts_id(self):
        pass

    @property
    @abstractmethod
    def X_full(self):
        pass

    @property
    @abstractmethod
    def Y_full(self):
        pass

    @property
    @abstractmethod
    def X_train(self):
        pass

    @property
    @abstractmethod
    def Y_train(self):
        pass

    @property
    @abstractmethod
    def X_calib(self):
        pass

    @property
    @abstractmethod
    def Y_calib(self):
        pass

    @property
    @abstractmethod
    def X_test(self):
        pass

    @property
    @abstractmethod
    def Y_test(self):
        pass

    @property
    @abstractmethod
    def no_test_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def no_of_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def first_prediction_step(self) -> int:
        pass

    @property
    @abstractmethod
    def no_calib_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def has_calib_set(self):
        pass

    @property
    @abstractmethod
    def calib_step(self):
        pass

    @property
    @abstractmethod
    def test_step(self):
        pass

    @property
    @abstractmethod
    def no_x_features(self):
        pass


class ChronoSplittedTsDataset(TsDataset):
    def __init__(self, ts_id: str, X, Y, test_step: int, calib_step: Optional[int] = None, normalize=True):
        super().__init__()
        self._ts_id = ts_id
        self._X = X
        self._Y = Y.view(Y.shape[0], -1)
        self._test_step = test_step
        self._calib_step = calib_step
        if calib_step is not None and test_step <= calib_step:
            raise ValueError("Calibration must be before training!")
        if test_step is None or test_step >= X.shape[0]:
            raise ValueError("Test Step must be defnied smaller than actual datset!")
        if normalize:
            self._X, self._X_means, self._X_stds = self._normalize(self._X)
            self._Y, self._Y_means, self._Y_stds = self._normalize(self._Y)
        else:
            self._X_means, self._X_stds = 0, 1.0
            self._Y_means, self._Y_stds = 0, 1.0

    def global_normalize(self, X_mean, X_std, Y_mean, Y_std):
            self._X = (self._X - X_mean) / X_std
            self._Y = (self._Y - Y_mean) / Y_std
            self._X_means, self._X_stds = X_mean, X_std
            self._Y_means, self._Y_stds = Y_mean, Y_std

    @property
    def X_normalize_props(self) -> Tuple[float, float]:
        return self._X_means, self._X_stds

    @property
    def Y_normalize_props(self) -> Tuple[float, float]:
        return self._Y_means, self._Y_stds

    @property
    def Y_std(self):
        return torch.std(self.Y_train)

    @property
    def ts_id(self):
        return self._ts_id

    @property
    def X_full(self):
        return self._X

    @property
    def Y_full(self):
        return self._Y

    @property
    def X_train(self):
        return self._X[:self.calib_step if self.has_calib_set else self.test_step]

    @property
    def Y_train(self):
        return self._Y[:self.calib_step if self.has_calib_set else self.test_step]

    @property
    def has_calib_set(self):
        return self.calib_step is not None

    @property
    def X_calib(self):
        if not self.has_calib_set:
            raise ValueError("No calibration set in this data!")
        return self._X[self.calib_step:self.test_step]

    @property
    def Y_calib(self):
        if not self.has_calib_set:
            raise ValueError("No calibration set in this data!")
        return self._Y[self.calib_step:self.test_step]

    @property
    def X_test(self):
        return self._X[self.test_step:]

    @property
    def Y_test(self):
        return self._Y[self.test_step:]

    @property
    def no_train_steps(self) -> int:
        return self.Y_train.shape[0]

    @property
    def no_calib_steps(self) -> int:
        return self.Y_calib.shape[0] if self.has_calib_set else 0

    @property
    def no_test_steps(self) -> int:
        return self.Y_test.shape[0]

    @property
    def no_of_steps(self) -> int:
        return self.Y_full.shape[0]

    @property
    def first_prediction_step(self) -> int:
        return self.calib_step if self.has_calib_set else self.test_step

    @property
    def calib_step(self):
        return self._calib_step

    @property
    def test_step(self):
        return self._test_step

    @property
    def no_x_features(self):
        return self.X_full.shape[1]


class BoostrapEnsembleTsDataset(TsDataset):
    """
    Implemented for the boostrap version of Xu et al. 2022 where train and calib data is shared
    by using an ensemble and the "left out" points as calibration points
    """
    def __init__(self, ts_id: str, X, Y, test_step: int, normalize=True):
        super().__init__()
        self._ts_id = ts_id
        self._X = X
        self._Y = Y.view(Y.shape[0], -1)
        self._test_step = test_step
        if test_step is None or test_step >= X.shape[0]:
            raise ValueError("Test Step must be defnied smaller than actual dataset!")
        if normalize:
            self._X, self._X_means, self._X_stds = self._normalize(self._X)
            self._Y, self._Y_means, self._Y_stds = self._normalize(self._Y)
        else:
            self._X_means, self._X_stds = 0, 1.0
            self._Y_means, self._Y_stds = 0, 1.0

    @property
    def X_normalize_props(self) -> Tuple[float, float]:
        return self._X_means, self._X_stds

    @property
    def Y_normalize_props(self) -> Tuple[float, float]:
        return self._Y_means, self._Y_stds

    @property
    def ts_id(self):
        return self._ts_id

    @property
    def X_full(self):
        return self._X

    @property
    def Y_full(self):
        return self._Y

    @property
    def X_train(self):
        return self._X[: self.test_step]

    @property
    def Y_train(self):
        return self._Y[: self.test_step]

    @property
    def X_calib(self):
        return self.X_train

    @property
    def Y_calib(self):
        return self.Y_train

    @property
    def X_test(self):
        return self._X[self.test_step:]

    @property
    def Y_test(self):
        return self._Y[self.test_step:]

    @property
    def no_test_steps(self) -> int:
        return self.Y_test.shape[0]

    @property
    def no_of_steps(self) -> int:
        return self.Y_full.shape[0]

    @property
    def first_prediction_step(self) -> int:
        return 0

    @property
    def no_calib_steps(self) -> int:
        return self.Y_calib.shape[0]

    @property
    def has_calib_set(self):
        return True

    @property
    def calib_step(self):
        return 0

    @property
    def test_step(self):
        return self._test_step

    @property
    def no_x_features(self):
        return self.X_full.shape[1]


class SimpleTsDataset:
    """
    Deprecated
    """

    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y


class HydroDataset(ChronoSplittedTsDataset):
    """Dataset for the hydrology application."""
    def __init__(self, ts_id: str, X, Y, test_step: int, static_attribute_indices: list[int],
                 static_attribute_norm_param,
                 calib_step: Optional[int] = None, normalize=True):

        self._static_attribute_indices = static_attribute_indices
        self._static_attribute_norm_param = static_attribute_norm_param
        # If there are no static attributes, we can use the default normalization from super.
        default_normalization = normalize and not static_attribute_indices
        super().__init__(ts_id, X, Y, test_step, calib_step, normalize=default_normalization)

        if normalize and static_attribute_indices:
            # Possible but not needed.
            raise NotImplementedError()

    @property
    def X_normalize_props(self) -> Tuple[float, float]:
        return self._X_means, self._X_stds

    @property
    def static_normalize_props(self):
        return self._static_attribute_norm_param

    def global_normalize(self, X_mean, X_std, Y_mean, Y_std):        
        self._Y = (self._Y - Y_mean) / Y_std
        self._Y_means, self._Y_stds = Y_mean, Y_std
        
        # Statics are normalized in data loading
        dynamic_indices = [i for i in range(self._X.shape[1]) if i not in self._static_attribute_indices]
        self._X[:, dynamic_indices] = (self._X[:, dynamic_indices] - X_mean[dynamic_indices]) / X_std[dynamic_indices]
        
        self._X_means, self._X_stds = X_mean, X_std
        self._X_means[self._static_attribute_indices], self._X_stds[self._static_attribute_indices] = 0.0, 1.0


        
