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



---

## Key Components

### Data Pipeline (src)

The data pipeline provides a structured framework for retrieving and processing aircraft images data.

#### Key Features:
 - Retrieval Engine: Fetches flight data using time intervals and query functions
 - Pipeline Modules: Specialized pipelines for different data types:
   - FlightsPipeline: Retrieves flight data with metadata
   - StateVectorPipeline: Retrieves state vector data (position, velocity, etc.)
 - Query Modules: SQL query generators for different data types
 - Transformation Modules: Data preprocessing utilities

### Model Tuning
  The state prediction component uses machine learning to predict future aircraft states based on historical trajectory data.

#### Models Implemented:

 - Transformer: Attention-based sequence model for capturing complex temporal dependencies
 - LSTM: Long Short-Term Memory network for sequential data
 - FFNN: Feed-Forward Neural Network for simpler prediction tasks
 - XGBoost: Gradient boosting for tabular data with engineered features
 - Kalman Filter: Traditional state estimation approach


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

 - Dynamic Time Warping (DTW): Algorithm for measuring similarity between temporal sequences
 - K-means Clustering: Unsupervised clustering to identify natural groupings of flight patterns
 - Prototype Matching: Comparison of flight patterns against predefined prototypes

#### Features:

 - Support for multi-dimensional DTW to compare multiple flight attributes
 - Prototype-based classification for known flight categories
 - CNN-based classification after unsupervised labeling

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




