import numpy as np

from utilities.Utilities import Utilities
from algorithms.initialization.Random import Random
from algorithms.initialization.InitializationMethod import InitializationMethod


class KMeans:
    def __init__(self, k: int, epsilon: float, initialization_method: InitializationMethod = Random,
                 initial_state: np.array = None):
        """
        A KMeans transformer.

        Parameters
        ----------
        k: int
             The number of clusters to fit

        epsilon: float
            The threshold for convergence

        initialization_method: InitializationMethod
            The centroid initialization method (defaults to Random initialization)

        initial_state: np.array
            An optional parameter to set the initial centroids of the clusters (defaults to None)

        Returns
        -------
        A KMeans transformer
        """

        # Define hyper parameters
        self._k = k
        self._epsilon = epsilon

        # Initialize algorithm properties
        self._data = None
        self._prev_centroids = None
        self._clusters = None
        self.centroids = initial_state
        self.cluster_assignment = None

        # Define initialization method
        self._initialization_method = initialization_method

    def _has_converged(self) -> bool:
        """
        Determine if the KMeans algorithm has converged to a minima by considering the change in objective function
        (specifically, comparing this change to the epsilon hyperparameter).

        Returns
        -------
        A boolean representing if the algorithm has converged according to the provided epsilon hyperparameter
        """

        sum_of_differences = 0

        # Sum the changes in cluster centroids between the previous and current iterations
        for cluster_id in range(self._k):
            sum_of_differences += Utilities.lp_norm(
                self.centroids[cluster_id] - self._prev_centroids[cluster_id]) ** 2

        # Check if the difference is small enough to consider the data converged
        return sum_of_differences <= self._epsilon

    def fit(self, data: np.array):
        """
        Fit K clusters to data using the KMeans algorithm.

        Parameters
        ----------
        data: np.array
             The data to fit clusters for to
        """

        self._data = data

        # If the representatives are not initialized, initialize them to a random state
        if self.centroids is None:
            self.centroids = self._initialization_method.initialize(data, self._k)

        # Initialize the previous centroids
        self._prev_centroids = [[np.inf] * data.shape[1]] * self._k

        while not self._has_converged():
            # Re-initialize cluster assignments to be empty
            self._clusters = {cluster_id: [] for cluster_id in range(self._k)}

            for index, value in enumerate(data):
                # Identify closest representative
                distances = np.apply_along_axis(
                    lambda representative: Utilities.lp_distance(value, representative, 2),
                    1,
                    self.centroids
                )

                # Assign point to associated cluster
                self._clusters[np.argmin(distances)].append((index, value))

            for cluster_id in range(self._k):
                # Update the previous centroid to be current
                self._prev_centroids[cluster_id] = self.centroids[cluster_id].copy()

                # Get all points that have been assigned to the current cluster
                points_in_cluster = [value for _, value in self._clusters[cluster_id]]

                # Compute the new centroids
                if len(points_in_cluster) == 0:
                    new_centroid = self._prev_centroids[cluster_id]
                else:
                    new_centroid = np.mean(points_in_cluster, axis=0)

                # Update the current centroids
                self.centroids[cluster_id] = new_centroid

        self.cluster_assignment = np.zeros(len(data))

        # Extract final cluster assignments from clusters object
        for cluster_id in self._clusters.keys():
            for point_index, _ in self._clusters[cluster_id]:
                self.cluster_assignment[point_index] = cluster_id

        return self.cluster_assignment
