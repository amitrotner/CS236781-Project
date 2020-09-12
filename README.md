# APPLYING DEEP-CLUSTERING APPROACHES TO CREATE MEANINGFUL REPRESENTATIONS OF ECG BEAT MORPHOLOGY #

## This project is currently ongoing - more results to come ## 

## Introduction ##

This is a PyTorch implementation of our Deep Learning on Computational Accelerators course project detailed in `Report.pdf`

by:
| Name            |E-mail                              |
| :-------------  |:---------------                    |
| Amit Rotner     |amitrotner@campus.technion.ac.il    |
| Shaked Doron    |shaked.doron@campus.technion.ac.il  |


## Installation ##

1. Clone this repository.
2. Install and activate the environment using the following commands:
```
conda env create -f environment.yml
conda activate final_project
```

## Project Structure ##
    .
    ├── cs236781                                        # Helper files
    │   ├── plot.py                                     # Helper function to plot experiments results 
    │   └── unit.py                                     # Helper classes to represent the result of fitting a model
    ├── ptb.py                                          # The implementation of the CNN network for the PTB dataset
    ├── mit_bih.py                                      # The implementation of the CNN network for the MIT-BIH dataset
    ├── autoencoder.py                                  # The implementation of the Autoencoder network
    ├── clustering.py                                   # The implementation of the clustering layer, kmeans, target distribution, and clustering predictions calculation 
    ├── utils.py                                        # Helper class implementing metrics for clustering evaluation
    ├── training.py                                     # The implementation of the model training and testing functions
    ├── PTB classification with CNN.ipynb               # A Jupyter notebook to perform and display PTB classification using the CNN network
    ├── MIT-BIH classification with CNN.ipynb           # A Jupyter notebook to perform and display MIT-BIH classification using the CNN network
    ├── DCEC ptb.ipynb                                  # A Jupyter notebook to perform and display PTB classification using the DCEC network
    ├── DCEC mit-bih.ipynb                              # A Jupyter notebook to perform and display MIT-BIH classification using the DCEC network
    ├── Experiments.ipynb                               # A Jupyter notebook to perform DCEC experiments varying on d
    ├── Report.pdf                                      # Project report
    ├── environment.yml
    └── README.md


## Reproducing results ##

1. Download datasets from: https://drive.google.com/drive/folders/1fefvwQfyTafnq0rybXCWT9wmSElzP58A?usp=sharing and place it in ./data/ folder.
2. Run the relevant Jupyter notebook.



