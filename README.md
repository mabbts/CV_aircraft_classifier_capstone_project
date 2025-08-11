# Capstone Project Team One
![Look-up-in-the-sky](docs/aircraft_banner.png)






```
.___  __ /\                        .__                      ._.
|   |/  |)/ ______ _____    ______ |  | _____    ____   ____| |
|   \   __\/  ___/ \__  \   \____ \|  | \__  \  /    \_/ __ \ |
|   ||  |  \___ \   / __ \_ |  |_> >  |__/ __ \|   |  \  ___/\|
|___||__| /____  > (____  / |   __/|____(____  /___|  /\___  >_
               \/       \/  |__|             \/     \/     \/\/
```

# Multi-class Fine-Grained Visual Classifier for Aircraft using Pytorch

This project is a comprehensive system for retrieving, processing, analyzing, and predicting aircraft PIL images from the "FGVCAircraft" dataset. It consists of three main components:

1. **Data Pipeline (``)**: A robust pipeline for retrieving and processing aircraft images data
2. **Model Tuning (``)**: Applying multiple customized and pretrained models with hyper-parameter searching for the best model
3. **Aircraft Classification ()**: Using the best model and parameters with deep learning techniques to apply classifying aircraft model variants

## Table of Contents
- [Project Overview](#project-overview)
- [Key Components](#key-components)
  - [Data Pipeline (src)](#data-pipeline-src)
  - [Model Tuning](#model-tuning)
  - [Aircraft Classification](#aircraft-classification)
- [Directory Structure](#directory-structure)
- [Example Usage](#example-usage)
- [Current Progress](#current-progress)
- [Next Steps](#next-steps)
- [Reference](#reference)
- [Contributing](#contributing)


## Project Overview

General Aviation flightline personnel are responsible for servicing a wide variety of aircraft, many of which have unique needs and sometimes unfamiliar configurations. Aircraft servicing mistakes have resulted in many expensive maintenance mishaps and, most seriously, in severalfatality accidents. One of the most critical and costly risks they face is misfueling - introducing the wrong type of fuel into the aircraft.
There are also cases that existing aircraft identification tools are often designed for aviation enthusiasts or casual plane spotters - not for operational use by flightline technicians. project was started to directly address the existing gaps. We aim to deliver an easy to use, real-time aircraft classification model that can run on an edge device for use by General Aviation flightline technicians. A simple image of an aircraft will return the relevant aircraft information along with an accuracy prediction, likely alternatives for comparison, instructions and safety information for routine maintenance and checks, as well as a lock out feature or supervisor override requirements for exceptions. This project is motivated by these key considerations:
- Risk Reduction: reducing the likelihood of costly or fatal servicing errors such as misfueling
- Environmental Efficiency: edge-based image classifiers are up to 10K times more efficient than ChatGPT 4o.
- Operational Reliability: No internet access required increases reliability.
---

## Key Components

### Data Pipeline (src)

We utilized the Oxford University benchmark dataset Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft). We developed code to check for the dataset locally, and if not available,the code downloads, decompresses, and organizes the dataset into a folder designated for the purpose. The benchmark dataset is well documented and contains 10,000 unlabeled images with accompanying text files that provide labels as needed for each image. In addition, the data is already split into Test, Train, and Validation. Data labels are provided for three levels of aircraft classification - manufacturer (30), family (70), and variant (100). 

#### Key Features:
 - FGVCAircraftDataset: A customized class to better retrieve and direct to the index of images labeled through several .txt files
 - Models: Specialized pipelines for different data types:
   - Customized CNN: Retrieves flight data with metadata
   - Pre-trained: Such as ResNet50 and EfficientNet
   - Attention-mechanism: such as Context-Aware Attentional Pooling (CAP) and Squeeze-and-Excitation (SE)
 - Utilities: SQL query generators for different data types
 - Transformation: We choose [Albumentationsx](https://albumentations.ai/docs/) over Pytorch default v2 due to efficiency and speed. We resized our images consistently to 224x224 shape throughout our training and evaluation.

### Model Tuning
  The model tuning component uses [Optuna](https://optuna.readthedocs.io/en/stable/) package to instantiate and optimize a study object by wrapping all the hyper-parameters we care such as .

#### Hyper-parameters considered:

 - Models: Attention-based sequence model for capturing complex temporal dependencies
 - loss_function: Long Short-Term Memory network for sequential data
 - optimizer: Feed-Forward Neural Network for simpler prediction tasks
 - scheduler: Gradient boosting for tabular data with engineered features
 - batch_size: Traditional state estimation approach
 - num_epochs: 


#### Key Features:

 - Model training and evaluation scripts
 - Hyperparameter optimization
 - Comprehensive metrics and visualization tools
 - Prediction capabilities for single flights or batches
 - Analysis tools for model performance and failure cases


### Aircraft Classification 

The flight classification component uses unsupervised learning techniques to identify and categorize flight patterns.
The flight classification component is primarily implemented in Jupyter notebooks for exploratory analysis and visualization. The main notebook is `notebooks/Base.ipynb`.

#### Key Techniques:
 - Transformer: Attention-based sequence model for capturing complex temporal dependencies
 - LSTM: Long Short-Term Memory network for sequential data
 - FFNN: Feed-Forward Neural Network for simpler prediction tasks
 - XGBoost: Gradient boosting for tabular data with engineered features
 - Kalman Filter: Traditional state estimation approach

 - Dynamic Time Warping (DTW): Algorithm for measuring similarity between temporal sequences
 - K-means Clustering: Unsupervised clustering to identify natural groupings of flight patterns
 - Prototype Matching: Comparison of flight patterns against predefined prototypes

#### Features:

 - Support for multi-dimensional DTW to compare multiple flight attributes
 - Prototype-based classification for known flight categories
 - CNN-based classification after unsupervised labeling - Model training and evaluation scripts
 - Hyperparameter optimization
 - Comprehensive metrics and visualization tools
 - Prediction capabilities for single flights or batches
 - Analysis tools for model performance and failure cases

---

## Directory Structure

Below is a high-level overview of the repository layout:
<!-- TREE START -->
```text
├── README.md                  # Project overview, setup instructions, usage
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignore model files, outputs, etc.
│
├── models/                    # Saved PyTorch models (.pth)
│   └── best_model_xxx.pth
│
├── docs/                      # Generated outputs (HTML, PNG, etc.)
│   ├── xxx.png
│   └── interactive_xxx.html
│
├── notebooks/                   # Notebooks (jupyter notebook scripts)
│   ├── Base.ipynb             # Main notebook 
│   ├── Viz.ipynb              # Hierarchical visualization of our datasets
│   └── hyperopt_optuna.ipynb  # hyper-parameter searching scripts
│
├── src/                       # Source code (modular Python scripts)
│   ├── data_utils.py
│   ├── models.py
│   ├── aircraft_utils.py
│   └── hyperopt.py
│
└── apps/                     # Final dashboard
    └── Dashboard.py

```
---
### Interactive .html Link to Visualization
[Click to Open]( https://mabbts.github.io/CV_aircraft_classifier_capstone_project/)


## Example Usage

### Data Pipeline

1. Clone the repo:
```
git clone https://github.com/mabbts/CV_aircraft_classifier_capstone_project.git

cd CV_aircraft_classifier_capstone_project
```
2. (Optional): Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```
3. Install requirements:
```
pip install -r requirements.txt
``` 
4. Activate GPU support on PC:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
5. Activate GPU support on Mac M-series chips:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```


### Model Tuning

All notebooks and scripts assume the repository root as the working directory.

The hyper-parameters searching is implemented through Jupyter notebooks in `notebooks/hyperopt_optuna.ipynb`. It takes more than 12 hours to execute finding multiple hyper-parameters.

### Aircraft Classification

All notebooks and scripts assume the repository root as the working directory.

The aircraft classification component is primarily implemented in Jupyter notebooks for exploratory analysis and visualization. The main notebook is `notebooks/Base.ipynb`, and `notebooks/Viz.ipynb` for visualization.

To run the final dashboard `apps/Dashboard.py`

```
python Dashboard.py
```
And you should be able to see the dashboard such like:

![Dashboard-sample](docs/dashboard.PNG)



---

## Current Progress

Data Pipeline: Fully implemented and tested for both flight data and state vector data

Model Turning:
 - Implemented multiple model architectures (Transformer, LSTM, XGBoost)
 - Created comprehensive evaluation framework
 - Analyzed model performance and failure cases
  
Aircraft Classification:
 - Implemented DTW-based similarity measurement
 - Created K-means clustering for flight pattern identification
 - Developed prototype matching for classification
 - Explored CNN-based classification after unsupervised labeling
  
---

## Next Steps

Data Pipeline:

 - Support for additional data sources
 - Real-time data streaming capabilities
Model Tuning:

 - Multi-modal prediction incorporating weather data
 - Uncertainty quantification in predictions
 - Ensemble methods combining multiple model types
  
Aircraft Classification:

 - Integration of supervised learning with domain expert labels
 - Anomaly detection for unusual flight patterns
 - Real-time classification capabilities

## Reference
---
The following resources were consulted during the development of this project:

- **Performance Analysis of Deep Learning Algorithms Implemented Using PyTorch in Image Recognition**
[Open Access - ScienceDirect]( https://www.sciencedirect.com/science/article/pii/S1877050924028084)

- **Benchmarking Deep Learning Models on NVIDIA Jetson Nano forcReal-Time Systems: An Empirical Investigation**
[Open Access - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050925010178)

- **Aircraft-Type Classification Using Deep Learning Algorithms**
[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10705261)

- **ANALYSIS OF EFFECTIVE IMAGE PROCESSING METRICS ON RASPBERRY PI AND NVIDIA JETSON NANO**
[ResearchGate](https://www.researchgate.net/publication/387534040_ANALYSIS_OF_EFFECTIVE_IMAGE_PROCESSING_METRICS_ON_RASPBERRY_PI_AND_NVIDIA_JETSON_NANO)

- **General Aviation Aircraft Identification at Non-Towered Airports Using a Two-Step Computer Vision-Based Approach**
[IEEE Xplore](https://ieeexplore.ieee.org/document/9770072)

## Contributing
This project is part of a course requirement, but feedback, suggestions, and ideas are welcome! Feel free to open issues or submit pull requests if you have improvements to suggest.

 - Issues: For bug reports or feature requests
  
 - Pull Requests: We welcome code contributions—please be sure to include clear descriptions and testing steps




