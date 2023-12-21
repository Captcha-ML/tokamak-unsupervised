# QCE Plasma regime investigation

Welcome to our project! This guide will help you get started with running the machine learning models in this repository.

## Getting Started

There are two main Jupyter notebooks in this project, each serving a different purpose:

1. **Run.ipynb**:
   - Use this file if you want to dive deep into all implementations and the reasoning behind them. It's a comprehensive notebook that covers the entire workflow from data preprocessing to model training and evaluation.

2. **Predict.ipynb**:
   - This file is for those who want to directly run the supervised model. It's streamlined for quick execution and getting predictions from the model without delving into the underlying details.

## Prerequisites

Before running either of the Jupyter notebooks, please ensure the following steps are completed:

a. **Data Download**:
   - Download the necessary datasets (both `.parquet` and `.csv` files) into a folder named `QCEH_data`. This data is crucial for the execution of the notebooks.

b. **Precomputed Data**:
   - Download the computationally expensive data from the `precomputed_data` file. This data is used for certain parts of the analysis to save on computation time.

## Execution

After setting up the data, you can simply open and run the Jupyter notebooks:

- Run.ipynb
- Predict.ipynb

Please note that some cells in the notebooks might execute faster than others. The entire process may take approximately 1-2 hours of computation time, depending on your system's capabilities.

## Additional Information

- Ensure that your Python environment has all the required dependencies installed. Refer to the `requirements.txt` file for a list of necessary packages.
- It's recommended to run these notebooks in a virtual environment to avoid dependency conflicts with other Python projects.

Happy Analyzing!
