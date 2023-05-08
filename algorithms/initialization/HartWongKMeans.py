import numpy as np
from algorithms.initialization.InitializationMethod import InitializationMethod


class HartWongKMeans(InitializationMethod):
    @classmethod
    def initialize(cls, data: np.array, k: int) -> np.array:
        """
        The Hartigan Wong initialization method which randomly assigns every data point to a cluster
        and then the uses the mean of these assignments to define initial cluster centroids

        Parameters
        ----------
        data: np.array
            The underlying data set

        k: int
            The number of centroids to initialize

        Returns
        -------
        A numpy array containing k initialized centroids
        """
        
        # Get the number of attributes represented in these data
        num_attributes = data.shape[1]

        # Initialize centroids to be all zeros
        centroids = np.zeros((k, num_attributes))

        # Create array to store random cluster assignments
        cluster_assignments = {cluster_id: [] for cluster_id in range(k)}

        # Randomly assign each point to a cluster
        for point in data:
            cluster_assignments[np.random.randint(k)].append(point)

        # Compute centroids of random clusters
        for cluster_id in range(k):
            # Get all points that have been assigned to the current cluster
            points_in_cluster = [value for _, value in cluster_assignments[cluster_id]]
            
            # Compute the new centroids
            centroids[cluster_id] = np.mean(points_in_cluster, axis=0)

        return centroids
