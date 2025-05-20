"""
One-shot Handwritten Character Classifier using Bayesian Program Learning (BPL)

This classifier uses modified Hausdorff distance to compare shapes of characters.
"""

import os
import numpy as np
import imageio.v3 as iio
from scipy.spatial.distance import cdist
from typing import Callable, List, Tuple, Union


class OneShotClassifier:
    """
    A class to handle one-shot handwritten character classification.

    Features:
    - Classification based on modified Hausdorff distance
    - Supports multiple runs for evaluation
    - Modular and object-oriented structure
    - Configurable via constructor parameters

    Attributes:
        nrun (int): Number of classification runs to perform
        script_dir (str): Directory containing this script
        all_runs_dir (str): Base directory for all run data
        label_filename (str): Name of the file containing class labels
    """

    def __init__(
        self,
        nrun: int = 20,
        script_dir: str = None,
        all_runs_dir: str = "all_runs",
        label_filename: str = "class_labels.txt"
    ):
        """
        Initialize the classifier with parameters

        Args:
            nrun (int): Number of classification runs to perform
            script_dir (str): Directory containing this script (optional)
            all_runs_dir (str): Base directory for run data
            label_filename (str): Name of file containing class labels

        Raises:
            ValueError: If provided script_dir does not exist
        """
        self.nrun = nrun
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.all_runs_dir = os.path.join(self.script_dir, all_runs_dir)
        self.label_filename = label_filename

        if script_dir is not None:
            if not os.path.exists(script_dir):
                raise ValueError(
                    f"Provided script directory '{script_dir}' does not exist.")
            self.script_dir = script_dir

    def _load_image_points(self, filename: str) -> np.ndarray:
        """
        Load an image and return coordinates of black pixels.

        Args:
            filename (str): Absolute path to the image file

        Returns:
            np.ndarray: [n x 2] array of pixel coordinates
        """
        image = iio.imread(filename)
        bool_array = ~np.array(image, dtype=np.bool_)
        coords = np.array(np.nonzero(bool_array)).T
        return coords - coords.mean(axis=0)

    def _modified_hausdorff_distance(self, itemA: np.ndarray, itemB: np.ndarray) -> float:
        """
        Compute modified Hausdorff distance between two sets of coordinates.

        Args:
            itemA (np.ndarray): [n x 2] array of coordinates for image A
            itemB (np.ndarray): [m x 2] array of coordinates for image B

        Returns:
            float: Modified Hausdorff distance
        """
        distances = cdist(itemA, itemB)
        min_dist_A = distances.min(axis=1)
        min_dist_B = distances.min(axis=0)
        return max(np.mean(min_dist_A), np.mean(min_dist_B))

    def _classification_run(
        self,
        folder: str,
        f_load: Callable[[str], np.ndarray],
        f_cost: Callable[[np.ndarray, np.ndarray], float],
        ftype: str = "cost"
    ) -> float:
        """
        Run one classification experiment

        Args:
            folder (str): Directory containing images for this run
            f_load (Callable[[str], np.ndarray]): Function to load and process an image
            f_cost (Callable[[np.ndarray, np.ndarray], float]): Cost function comparing two processed images
            ftype (str): "cost" or "score" indicating which values are better

        Returns:
            float: Percentage error rate

        Raises:
            ValueError: If folder does not exist
            FileNotFoundError: If labels file is missing
        """
        if not os.path.exists(os.path.join(self.all_runs_dir, folder)):
            raise ValueError(
                f"The folder {folder} does not exist in the runs directory.")

        labels_path = os.path.join(
            self.all_runs_dir, folder, self.label_filename)

        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file {labels_path} not found.")

        with open(labels_path) as label_file:
            pairs = [line.strip().split() for line in label_file.readlines()]

        test_files, train_answers = zip(*pairs)
        n_test = len(test_files)
        n_train = len(train_answers)

        # Load and process images
        try:
            test_images = [
                f_load(os.path.join(self.all_runs_dir, f))
                for f in sorted(test_files)
            ]
            train_images = [
                f_load(os.path.join(self.all_runs_dir, f))
                for f in sorted(train_answers)
            ]

            # Calculate cost matrix
            cost_matrix = np.zeros((n_test, n_train))

            for i, test_img in enumerate(test_images):
                for j, train_img in enumerate(train_images):
                    cost_matrix[i, j] = f_cost(test_img, train_img)

            if ftype == "cost":
                predicted_indices = np.argmin(cost_matrix, axis=1)
            elif ftype == "score":
                predicted_indices = np.argmax(cost_matrix, axis=1)
            else:
                raise ValueError("ftype must be 'cost' or 'score'")

            # Calculate error rate
            correct_predictions = sum(
                1 for i in range(n_test)
                if train_answers[i] == sorted(train_answers)[predicted_indices[i]]
            )

            return (n_test - correct_predictions) / n_test * 100

        except Exception as e:
            raise RuntimeError(
                f"Error during classification run: {str(e)}") from e

    def run_experiments(self):
        """
        Run all classification experiments and report average error rate
        """
        print("Running one-shot handwritten character classifier")
        error_rates = []

        for run in range(1, self.nrun + 1):
            folder_name = f"run{run:02d}"
            try:
                error_rate = self._classification_run(
                    folder_name,
                    self._load_image_points,
                    self._modified_hausdorff_distance,
                    "cost"
                )
                error_rates.append(error_rate)
                print(f"[INFO] Run {run:02d}: Error rate {error_rate:.1f}%")
            except Exception as e:
                print(f"[ERROR] Run {run:02d} failed: {str(e)}")

        average_error = np.mean(error_rates)
        print(
            f"\n[RESULT] Average error rate across {self.nrun} runs: {average_error:.1f}%")


if __name__ == "__main__":
    classifier = OneShotClassifier()
    classifier.run_experiments()
