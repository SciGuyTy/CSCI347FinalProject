import numpy as np


class InitializationMethod:
    @classmethod
    def initialize(cls, data: np.array, k: int) -> np.array:
        """
        An abstract method for initializing k centroids

        Parameters
        ----------
        data: np.array
            The underlying data set

        k: int
            The number of centroids to initialize

        Returns
        -------
        A numpy array containing the k initialized centroids
        """
        pass
