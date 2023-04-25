import numpy as np
from algorithms.initialization.InitializationMethod import InitializationMethod
from utilities.Utilities import Utilities


class KMeansPlusPlus(InitializationMethod):
    """
    KMeans++ initialization method that iteratively selects centroids according to a probability distribution that is
    weighted with respect the distances between each point and their nearest existing centroid
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

        # Randomly select the first centroid from the data points
        centroids = [data[np.random.randint(len(data))]]

        # Find remaining centroids
        for index in range(k - 1):
            distances = []

            for point in data:
                # Compute the euclidean distance to each centroid
                centroid_distances = np.apply_along_axis(
                    lambda centroid: Utilities.lp_distance(point, centroid, 2),
                    1,
                    centroids
                )

                # Identify the smallest distance (i.e., distance to the nearest centroid for the point)
                distances.append(min(centroid_distances))

            # Create an array of weighted probabilities that is proportional to the distance between each point and
            # their nearest centroid
            weighted_probabilities = np.multiply(distances, (1 / np.sum(distances)))

            # Select the next centroid according to the weighted probability distribution
            new_centroid = data[distances.index(np.random.choice(distances, 1, p=weighted_probabilities)[0])]
            centroids.append(new_centroid)

        return np.array(centroids)
