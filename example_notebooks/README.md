# Show case of using python DASK
This directory includes some example to speed up access MATS dataset and calculations using the data that resides from AWS S3.

## What is DASK

Dask is an open-source parallel computing framework designed to scale data analysis and machine learning workflows from a single machine to distributed clusters. It provides advanced parallelism for analytics, enabling efficient processing of large datasets by breaking tasks into smaller chunks that can be executed concurrently. Dask integrates seamlessly with familiar Python libraries like NumPy, pandas, and scikit-learn, making it an accessible tool for scaling Python-based computations without significant code rewrites.

### Costs for running DASK cluster in AWS

In the eu-north-1 region:

Scheduler: 1 vCPU, 4 GB memory = $0.05826/hour.

Workers: 5 workers, each with 4 vCPUs, 16 GB memory = $1.1652/hour.

Total cost: $1.22346/hour.

## Environment for running DASK 

Having the same environment on the client and the cluster is crucial when using Dask for the following reasons:

Compatibility: Ensures the same versions of libraries (e.g., pandas, NumPy, scikit-learn) are available, preventing errors caused by mismatched APIs or dependencies.

Serialization: Dask often serializes and transfers functions and data between the client and workers. Mismatched environments can lead to serialization errors or unexpected behavior.

Reproducibility: A consistent environment ensures that computations behave the same on the client and the cluster, making debugging and validation easier.

Performance: Matching versions can prevent inefficiencies from fallback behaviors or incompatible optimizations.

Inconsistent environments can result in runtime errors, subtle bugs, or inefficient execution.

### Environment in the examples

For the examples use python3.12 in a virtual environment with the dependencies described in the requirement*.txt files 

```bash
mkvirtualenv -p python3.12 example_notebooks
cd example_notebooks
pip install pip-tools
pip-sync requirements-dev.txt
```
