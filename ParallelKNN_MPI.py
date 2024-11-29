import numpy as np
from mpi4py import MPI
import pandas as pd
import heapq
import time
from sklearn.preprocessing import StandardScaler
import argparse
import os
import logging
from tqdm import tqdm


class ParallelKNN:
    def __init__(self, train_file, test_file):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.train_file = train_file
        self.test_file = test_file

    def load_data(self):
        """
        Load train and test datasets and normalize them.
        """
        if self.rank == 0:
            train_data = pd.read_csv(self.train_file, header=None).values
            test_data = pd.read_csv(self.test_file, header=None).values
            #print(f"[Process {self.rank}] Loaded Train Data Shape: {train_data.shape}")
            #print(f"[Process {self.rank}] Loaded Test Data Shape: {test_data.shape}")
            train_data = self.normalize_data(train_data)
            test_data = self.normalize_data(test_data)
        else:
            train_data = None
            test_data = None

        # Broadcast normalized datasets to all processes
        train_data = self.comm.bcast(train_data, root=0)
        test_data = self.comm.bcast(test_data, root=0)
        return train_data, test_data

    def normalize_data(self, data):
        """
        Normalize the dataset using standard scaling (Z-score normalization).
        """
        scaler = StandardScaler()
        features = data[:, :-1]
        labels = data[:, -1].reshape(-1, 1)
        scaled_features = scaler.fit_transform(features)
        return np.hstack((scaled_features, labels))

    def distribute_training_data(self, train_data):
        """
        Split the training data evenly across processes and synchronize chunk size printing.
        """
        num_samples = len(train_data)
        chunk_size = num_samples // self.size
        remainder = num_samples % self.size

        start_idx = self.rank * chunk_size + min(self.rank, remainder)
        end_idx = start_idx + chunk_size + (1 if self.rank < remainder else 0)
        local_train_data = train_data[start_idx:end_idx]

        # Only print chunk sizes during initial distribution
        if not hasattr(self, '_chunks_printed'):
            chunk_info = self.comm.gather(f"[Process {self.rank}] Training data chunk size: {local_train_data.shape}", root=0)
            if self.rank == 0:
                print("\nTraining Data Chunk Sizes:")
                print("\n".join(chunk_info))
            self._chunks_printed = True

        return local_train_data

    def compute_distances(self, train_data, test_instance):
        """
        Compute squared Euclidean distances between test instance and all rows in training data.
        """
        return np.linalg.norm(train_data[:, :-1] - test_instance[:-1], axis=1)

    def find_local_knn(self, local_train_data, test_instance, k):
        """
        Find the k-nearest neighbors for a test instance from local training data.
        """
        distances = self.compute_distances(local_train_data, test_instance)
        local_neighbors = [(distances[i], local_train_data[i, -1]) for i in range(len(distances))]
        return heapq.nsmallest(k, local_neighbors, key=lambda x: x[0])

    def gather_global_knn(self, local_neighbors, k):
        """
        Gather local k-nearest neighbors from all processes and determine global k-nearest neighbors.
        """
        all_neighbors = self.comm.gather(local_neighbors, root=0)
        if self.rank == 0:
            combined_neighbors = [neighbor for sublist in all_neighbors for neighbor in sublist]
            global_neighbors = heapq.nsmallest(k, combined_neighbors, key=lambda x: x[0])
            return global_neighbors
        return None

    def classify_instance(self, global_neighbors):
        """
        Classify a test instance using weighted voting.
        """
        weights = {}
        for distance, label in global_neighbors:
            weights[label] = weights.get(label, 0) + 1 / (distance + 1e-5)  # Weighted by inverse distance
        return max(weights, key=weights.get)  # Class with the highest weighted sum

    def classify(self, train_data, test_data, k):
        """
        Classify all test instances using the k-NN algorithm.
        """
        local_train_data = self.distribute_training_data(train_data)
        predictions = []
        
        if self.rank == 0:
            test_data_iter = tqdm(test_data, desc="Classifying", unit="instance")
        else:
            test_data_iter = test_data

        for test_instance in test_data_iter:
            local_neighbors = self.find_local_knn(local_train_data, test_instance, k)
            global_neighbors = self.gather_global_knn(local_neighbors, k)
            if self.rank == 0:
                predictions.append(self.classify_instance(global_neighbors))

        return predictions if self.rank == 0 else None

    def evaluate(self, predictions, test_data):
        """
        Evaluate the accuracy of the classifier on the test dataset.
        """
        true_labels = test_data[:, -1].astype(int)
        accuracy = np.mean(np.array(predictions) == true_labels) * 100
        return accuracy

    def run(self):
        """
        Run the k-NN algorithm, dynamically test multiple k values, and output results.
        """
        train_data, test_data = self.load_data()

        if self.rank == 0:
            # Print the number of processes
            print(f"Number of Processes: {self.size}")
            # Print the loaded data shapes
            print(f"[Process {self.rank}] Loaded Train Data Shape: {train_data.shape}")
            print(f"[Process {self.rank}] Loaded Test Data Shape: {test_data.shape}")
            
            

            logging.info(f"Training Dataset: {train_data.shape[0]} samples")
            logging.info(f"Testing Dataset: {test_data.shape[0]} samples\n")

            k_index = int(input("\nEnter k number of neighbors (e.g., 6): "))
            k_values = [2**(3 + i - 1) for i in range(1, k_index + 1)]
        else:
            k_values = None

        k_values = self.comm.bcast(k_values, root=0)
        
        # Distribute training data and print chunk sizes
        local_train_data = self.distribute_training_data(train_data)

        # Print test instance processing message after chunk sizes
        if self.rank == 0:
            print(f"\nProcessing {len(test_data)} test instances...")
            print(f"Processed {len(test_data)}/{len(test_data)} test instances\n")

        # Print table header on the master process
        if self.rank == 0:
            print("Results for k Nearest Neighbors:")
            print("+-------------------+------------------+--------------------------+")
            print("|      k Value      |    Accuracy (%)       |  Execution Time (s) |")
            print("+-------------------+------------------+--------------------------+\n")

        # Dynamically test each k value
        for k in k_values:
            if self.rank == 0:
                start_time = time.time()

            # Classify test data for current k value
            predictions = self.classify(train_data, test_data, k)

            if self.rank == 0:
                # Evaluate and print results dynamically
                accuracy = self.evaluate(predictions, test_data)
                end_time = time.time()
                print(f"\n--> K ={k:<17} | {accuracy:<16.2f} | {end_time - start_time:<19.4f} |\n")
            
        # Print table footer on the master process
        if self.rank == 0:
            print("\nThank you!\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel k-NN using MPI')
    parser.add_argument('--train_file', type=str, default='/Applications/PA/data/train.csv', help='Path to the training data file')
    parser.add_argument('--test_file', type=str, default='/Applications/PA/data/test.csv', help='Path to the testing data file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    setup_logging()
    args = parse_arguments()
    
    if not os.path.exists(args.train_file) or not os.path.exists(args.test_file):
        raise FileNotFoundError("Train or test file not found. Please check the file paths.")
    
    knn = ParallelKNN(args.train_file, args.test_file)
    knn.run()

if __name__ == "__main__":
    main()






