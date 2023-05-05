from collections import Counter

import numpy as np


class Utilities:
    @classmethod
    def standard_deviation(cls, x1: np.array):
        """
        Compute the standard deviation of a vector

        Parameters
        ----------
        x1: np.array
            The vector with which to compute the standard deviation for

        Returns
        -------
        A float representing the standard deviation
        """

        # Compute sample mean
        mean = x1.mean()

        # Compute the sum of squared mean differences
        sum_mean_difference = np.sum((x1 - mean) ** 2)

        # Compute and return the standard deviation
        return np.sqrt(sum_mean_difference / (len(x1) - 1))

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
        A float representing the L_p distance between vectors x1 and x2
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
    def contingency_table(cls, ground_truth: np.array, predicted: np.array) -> np.array:
        num_true_clusters = np.max(ground_truth) + 1

        # Create an empty contingency table whose dimensions are bounded by the number of ground truth clusters
        table = np.zeros((num_true_clusters, num_true_clusters))

        # Populate the table
        for index, expected in enumerate(ground_truth):
            predicted_class = int(predicted[index])

            # Wrap in try/catch to easily handle cases when there are more predicted clusters than true clusters
            try:
                table[predicted_class][expected] += 1
            except:
                pass

        return table

    @classmethod
    def precision(cls, contingency_table: np.array) -> dict[int, float]:
        """
        Compute the precision of a clustering

        Parameters
        ----------
        contingency_table: np.array
            The contingency table for the clustering

        Returns
        -------
        A dictionary containing the precision for each predicted cluster
        """

        # Create an empty dictionary with a space for each ground truth class
        precision_dict = dict.fromkeys(range(np.shape(contingency_table)[0]), 0)

        for index, predicted_cluster in enumerate(contingency_table):
            # Get the number of predictions from the dominant class in the cluster
            num_in_dominant = np.max(predicted_cluster)

            # Get the number of predictions in the cluster
            num_in_cluster = np.sum(predicted_cluster)

            # Compute cluster precision (num of dominant / num in cluster)
            precision_dict[index] = (num_in_dominant / num_in_cluster) if num_in_cluster > 0 else 0

        return precision_dict

    @classmethod
    def recall(cls, contingency_table: np.array) -> dict[int, float]:
        """
        Compute the recall of a clustering

        Parameters
        ----------
        contingency_table: np.array
            The contingency table for the clustering

        Returns
        -------
        A dictionary containing the recall for each predicted cluster
        """

        # Create an empty dictionary with a space for each ground truth class
        recall_dict = dict.fromkeys(range(np.shape(contingency_table)[0]), 0)

        for index, predicted_cluster in enumerate(contingency_table):
            # Get the number of predictions from the dominant class in the cluster
            num_in_dominant = np.max(predicted_cluster)

            # Get the number of predictions in the dominant class in the true cluster
            num_total_dominant = np.sum(contingency_table.T[np.argmax(predicted_cluster)])

            # Compute cluster precision (num of dominant / num of total dominant)
            recall_dict[index] = num_in_dominant / num_total_dominant

        return recall_dict

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

        # Create a contingency table for the predictions
        contingency_table = cls.contingency_table(ground_truth, predictions)

        # Compute the precision and recall for each predicted cluster
        precision = cls.precision(contingency_table)
        recall = cls.recall(contingency_table)

        f_score = 0
        true_classes = np.unique(ground_truth)

        # Compute the F-beta score for each predicted class to compute harmonic average
        for true_class in true_classes:
            if precision[true_class] + recall[true_class] == 0:
                f_score += 0
            else:
                f_score += beta * ((precision[true_class] * recall[true_class]) / (
                            precision[true_class] + recall[true_class]))

        # Compute and return the F_beta score
        return f_score / len(true_classes)
