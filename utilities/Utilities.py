from collections import Counter

import numpy as np


class Utilities:
    @classmethod
    def lp_norm(cls, x1: np.array, deg: int = 2):
        norm = 0

        for element in x1:
            norm += element ** deg

        return norm ** (1 / deg)

    @classmethod
    def lp_distance(cls, x1: np.array, x2: np.array, deg: int = 2) -> float:
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 do not have the same shape")

        distance = 0

        for index in range(len(x1)):
            distance += (x1[index] - x2[index]) ** deg

        return distance ** (1 / deg)

    @classmethod
    def precision(cls, ground_truth: np.array, predictions: np.array) -> tuple[
        dict[int, float],
        float
    ]:
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
        # Dict to store precision for each cluster
        cluster_recall = {}

        # Compute the precision for each cluster
        for cluster_id in np.unique(predictions):
            # Get the ground truth values corresponding to the points in the predicted cluster
            predicted_points = [ground_truth[index] for index, assignment in enumerate(predictions) if
                                assignment == cluster_id]

            # Get the number of predicted points belonging to the dominant ground-truth cluster
            num_dominant_cluster = Counter(predicted_points).most_common()[0][1]

            num_ground_truth_cluster = len([element for element in ground_truth if element == cluster_id])

            # Get the number of predictions in the predicted cluster
            # num_predictions = len(predicted_points)

            recall = 0

            if num_ground_truth_cluster != 0:
                recall = num_dominant_cluster / num_ground_truth_cluster

            # Compute and return the precision
            cluster_recall[cluster_id] = recall

        # Compute the average precision across all clusters
        average_precision = np.average(list(cluster_recall.values()))

        return cluster_recall, average_precision

    @classmethod
    def f1_score(cls, prediction: np.array, ground_truth: np.array, ratio: int = 2) -> float:
        if prediction.shape != ground_truth.shape:
            raise ValueError("prediction and ground_truth labels do not have the same shape")

        precision = cls.precision(prediction, ground_truth)[1]
        recall = cls.recall(prediction, ground_truth)[1]

        print(precision, recall)

        return ratio * ((precision * recall) / (precision + recall))
