from collections import Counter

import numpy as np


class Utilities:
    @classmethod
    def lp_norm(cls, x1: np.array, p: int = 2) -> float:
        """
        Compute the L_p norm for a vector

        Parameters
        ----------
        x1: np.array
            The vector with which to compute the l_p norm for

        p: int
            The degree p of the norm (defaults to 2)

        Returns
        -------
        A float representing the L_p norm
        """

        norm = 0

        # Sum the 'p'th power of each element in the vector
        for element in x1:
            norm += element ** p

        # Compute and return the L_p norm
        return norm ** (1 / p)

    @classmethod
    def lp_distance(cls, x1: np.array, x2: np.array, p: int = 2) -> float:
        """
        Compute the L_p distance between two vectors

        Parameters
        ----------
        x1: np.array
            The first vector

        x2: np.array
            The second vector

        p: int
            The degree p of the distance computation (defaults to 2)

        Returns
        -------
        A float representing the L_p distasnce between vectors x1 and x2
        """

        # Validate the shape of the input vectors
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 do not have the same shape")

        distance = 0

        # Sum the 'p'th power of the pair-wise differences for each element in the vectors
        for index in range(len(x1)):
            distance += (x1[index] - x2[index]) ** p

        # Compute and return the L_p distance
        return distance ** (1 / p)

    @classmethod
    def precision(cls, ground_truth: np.array, predictions: np.array) -> tuple[
        dict[int, float],
        float
    ]:
        """
        Compute the precision of a clustering

        Parameters
        ----------
        ground_truth: np.array
            The ground truth cluster assignment

        predictions: np.array
            The predicted cluster assignment

        Returns
        -------
        A tuple whose first element is a dictionary containing the precision for each predicted cluster, and whose
        second element is a float representing the average precision across all predicted clusters
        """

        # Dict to store precision for each cluster
        cluster_precisions = {}

        # Compute the precision for each cluster
        for cluster_id in np.unique(predictions):
            # Get the ground truth values corresponding to the points in the predicted cluster
            predicted_points = [ground_truth[index] for index, assignment in enumerate(predictions) if
                                assignment == cluster_id]

            # Get the number of points belonging to the dominant ground-truth cluster
            num_dominant_cluster = Counter(predicted_points).most_common()[0][1]

            # Get the number of predictions in the predicted cluster
            num_predictions = len(predicted_points)

            # Compute and return the precision
            cluster_precisions[cluster_id] = num_dominant_cluster / num_predictions

        # Compute the average precision across all clusters
        average_precision = np.average(list(cluster_precisions.values()))

        return cluster_precisions, average_precision

    @classmethod
    def recall(cls, ground_truth: np.array, predictions: np.array) -> tuple[
        dict[int, float],
        float
    ]:
        """
        Compute the recall of a clustering

        Parameters
        ----------
        ground_truth: np.array
            The ground truth cluster assignment

        predictions: np.array
            The predicted cluster assignment

        Returns
        -------
        A tuple whose first element is a dictionary containing the recall for each predicted cluster, and whose
        second element is a float representing the average recall across all predicted clusters
        """

        # Dict to store precision for each cluster
        cluster_recall = {}

        # Compute the precision for each cluster
        for cluster_id in np.unique(predictions):
            # Get the ground truth values corresponding to the points in the predicted cluster
            predicted_points = [ground_truth[index] for index, assignment in enumerate(predictions) if
                                assignment == cluster_id]

            # Get the number of predicted points belonging to the dominant ground-truth cluster
            num_predicted_dominant_cluster = Counter(predicted_points).most_common()[0][1]

            # Get the number of total points belonging to the dominant ground-truth cluster
            num_dominant_cluster = len([element for element in ground_truth if element == cluster_id])

            # Compute the recall
            cluster_recall[cluster_id] = num_predicted_dominant_cluster / num_dominant_cluster

        # Compute the average precision across all clusters
        average_precision = np.average(list(cluster_recall.values()))

        return cluster_recall, average_precision

    @classmethod
    def f_score(cls, ground_truth: np.array, predictions: np.array, beta: int = 2) -> float:
        """
        Compute the F_beta score for a clustering

        Parameters
        ----------
        ground_truth: np.array
            The ground truth cluster assignment

        predictions: np.array
            The predicted cluster assignment

        beta: int
            The beta value that determines the ratio of precision and recall in the resulting F score (defaults to 2)

        Returns
        -------
        A float representing the F_beta score
        """

        # Validate the shape of the label assignments
        if predictions.shape != ground_truth.shape:
            raise ValueError("prediction and ground_truth labels do not have the same shape")

        # Compute the average precision and recall for these data
        precision = cls.precision(predictions, ground_truth)[1]
        recall = cls.recall(predictions, ground_truth)[1]

        # Compute and return the F_beta score
        return beta * ((precision * recall) / (precision + recall))
