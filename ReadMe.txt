README: Parallel k-NN Algorithm Using MPI

---

Project Title
Parallel k-Nearest Neighbors (k-NN) Algorithm Using MPI

---

Description
This project implements a parallelized version of the k-Nearest Neighbors (k-NN) algorithm using the Message Passing Interface (MPI). It is designed to classify large datasets efficiently by distributing workloads across multiple processes.

---

Features
- Scalability: Efficient handling of large datasets through workload distribution.
- Parallelization: Speedup of computations via distributed distance calculations and result aggregation.
- Accuracy: Maintains prediction accuracy comparable to sequential implementations.
- Dynamic (k)-values: User-specified (k)-values tested dynamically.

---

Prerequisites
1. System Requirements:
   - Linux, macOS, or Windows with MPI support.
   - Python 3.x installed.
   - MPI runtime environment (e.g., `mpich` or `OpenMPI`).

2. **Required Python Libraries:**
   - `mpi4py`
   - `numpy`
   - `pandas`
   - `scikit-learn`

3. **Datasets:**
   - Place the training and testing datasets in the directory `/Applications/PA/data/`:
     - `train.csv`: Training dataset
     - `test.csv`: Test dataset

---

Installation Instructions
1. Install MPI on your system:
   - For Linux/macOS:
     ```bash
     sudo apt-get install mpich  # For Ubuntu
     brew install mpich          # For macOS
     ```
   - For Windows, download and install MPI from [MS-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

2. Install Python dependencies:
   ```bash
   pip install mpi4py numpy pandas scikit-learn
   ```

3. Verify MPI installation:
   ```bash
   mpirun --version
   ```

---

Usage Instructions

1. Save the Python script as `ParallelKNN_MPI.py`.
2. Place the datasets (`train.csv` and `test.csv`) in the directory `/Applications/PA/data/`.
3. Run the script using `mpirun`:
   ```bash
   mpirun -np <number_of_processes> python ParallelKNN_MPI.py
   ```
   Example:
   ```bash
   mpirun -np 4 python ParallelKNN_MPI.py
   ```

4. Follow the prompts:
   - Enter the maximum(k)-value index when prompted (e.g., 2 for ( k = 8, 16, 32, ...)).

---

Project Workflow:

1. Data Distribution:
   - Training data is split evenly among processes.
   - Each process receives a full copy of the test dataset.

2. Local Distance Computation:
   - Each process computes distances between test points and its local training data.

3. Local k-NN Selection:
   - Local (k)-nearest neighbors are selected using priority queues.

4. *Global Aggregation:
   - Local results are gathered by the master process, which determines the global (k)-nearest neighbors.

5. Classification:
   - Test points are classified using majority or weighted voting.

---

Expected Output
- **Training Data Chunk Sizes:** Displayed once after data distribution.
- **Testing Results:** Accuracy and execution time for each (k)-value, dynamically displayed.


Project Structure

```
ParallelKNN_MPI/
│
├── ParallelKNN_MPI.py     # Main Python script
├── /Applications/PA/data/ # Data directory
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset
```

---

Features and Algorithms
1. Algorithms:
   - Data Distribution
   - Local Distance Computation
   - Local k-NN Selection
   - Global Aggregation
   - Classification with Majority/Weighted Voting

2. Scalability:
   - Efficient handling of large datasets.
   - Supports both strong and weak scaling.

3. Improvements for Scalability:
   - Dynamic load balancing.
   - GPU acceleration for local distance computations.
   - Support for alternative distance metrics.

---

Future Improvements
- Use GPUs to accelerate distance computations.  
- Implement hybrid parallelism (MPI + threading).  
- Support alternative metrics like Manhattan or cosine distance.  
- Optimize communication with collective MPI operations.

---

Team Members
- Reddy Bhuvan Korlakunta (Team Leader)  
- Om Preetham Bandi  
- Thriveen Ullendula  
- Nandini Kodali  
- Sravyasri Virigineni  
- Sravani Jampani  

---

Acknowledgments
- Texas A&M University Corpus Christi  
- Course: COSC 6361.001, Fall 2024  
- Instructor: Dr. Minhua Huang

--- 