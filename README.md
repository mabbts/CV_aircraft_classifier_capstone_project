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

# Multi-class Visual Classifier for Aircraft

This project is a comprehensive system for retrieving, processing, analyzing, and predicting aircraft PIL images from the "FGVCAircraft". It consists of three main components:

1. **Data Pipeline (``)**: A robust pipeline for retrieving and processing flight data
2. **Model Trainng (``)**: Deep learning models for predicting future aircraft states
3. **Aircraft Classification ()**: Supervised learning techniques for classifying aircraft model variants

## Table of Contents
- [Project Overview](#project-overview)
- [Key Components](#key-components)
  - [Data Pipeline (src)](#data-pipeline-src)
  - [State Prediction](#state-prediction)
  - [Flight Classification (ENID)](#flight-classification-enid)
- [Directory Structure](#directory-structure)
- [Example Usage](#example-usage)
- [Current Progress](#current-progress)
- [Next Steps](#next-steps)
- [Contributing](#contributing)


## Project Overview



---

## Key Components

### Data Pipeline (src)

The data pipeline provides a structured framework for retrieving and processing flight data.

#### Key Features:


### State Prediction


#### Models Implemented:


#### Key Features:


### Flight Classification 



#### Key Techniques:


#### Features:


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
## Interactive .html Link to Visualization
[Click to Open]( https://mabbts.github.io/CV_aircraft_classifier_capstone_project/)


## Example Usage

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
6. Activate GPU support on Mac M-series chips:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
7. Run the Dashboard:
All notebooks and scripts assume the repository root as the working directory.
```
python Dashboard.py
```

### Data Pipeline



### State Prediction



### Flight Classification

The flight classification component is primarily implemented in Jupyter notebooks for exploratory analysis and visualization. The main notebook is `notebooks/Base.ipynb`.

---

## Current Progress


---

## Next Steps



---
The following resources were consulted during the development of this project:

**Performance Analysis of Deep Learning Algorithms Implemented Using PyTorch in Image Recognition**
[Open Access - ScienceDirect]( https://www.sciencedirect.com/science/article/pii/S1877050924028084)

**Benchmarking Deep Learning Models on NVIDIA Jetson Nano forcReal-Time Systems: An Empirical Investigation**
[Open Access - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050925010178)

**Aircraft-Type Classification Using Deep Learning Algorithms**
[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10705261)

**ANALYSIS OF EFFECTIVE IMAGE PROCESSING METRICS ON RASPBERRY PI AND NVIDIA JETSON NANO**
[ResearchGate](https://www.researchgate.net/publication/387534040_ANALYSIS_OF_EFFECTIVE_IMAGE_PROCESSING_METRICS_ON_RASPBERRY_PI_AND_NVIDIA_JETSON_NANO)

**General Aviation Aircraft Identification at Non-Towered Airports Using a Two-Step Computer Vision-Based Approach**
[IEEE Xplore](https://ieeexplore.ieee.org/document/9770072)

## Contributing





