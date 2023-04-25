import numpy as np
from algorithms.initialization.InitializationMethod import InitializationMethod


class Random(InitializationMethod):
    """
    Random initialization method that selects centroids according to a uniform random distribution
    """

    @classmethod
    def initialize(cls, data: np.array, k: int) -> np.array:
        """
        Initialize k centroids according to the KMeans++ initialization algorithm

        Parameters
        ----------
        data: np.array
            The underlying data set

        k: int
            The number of centroids to initialize

        Returns
        -------
            A numpy array containing K initialized centroids
        """

        # Get the number of attributes represented in these data
        num_attributes = data.shape[1]

        # Get the minimum and maximum values for each attributes
        data_range = (data.min(axis=0), data.max(axis=0))

        # Generate a nd-array array to represent the centroids
        centroids = np.zeros((k, num_attributes))

        # Populate each centroid, with each value of the centroid being a random uniform value
        # bounded by the minimum and maximum values for the respective attribute
        for index, _ in enumerate(centroids):
            for column in range(num_attributes):
                centroids[index][column] = np.random.uniform(data_range[0][column], data_range[1][column])

        return centroids
