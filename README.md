# Telco Customer Churn Prediction

This project predicts customer churn for a telecommunications company using machine learning. It is built with Python, utilizes the CatBoost algorithm for classification, and provides a user-friendly prediction interface via a Streamlit web app.

## Features
- Data preprocessing and feature engineering
- Model training with CatBoost
- Model evaluation and visualization
- Streamlit web interface for predictions
- Modular and maintainable code structure

## Project Structure
```
├── README.md
├── requirements.txt
├── environment.yml
├── streamlit_app.py
├── catboost_info/
├── config/
├── data/
├── models/
├── outputs/
└── src/
```
- **data/**: Raw datasets
- **models/**: Trained models and pipeline files
- **src/**: Source code (data processing, training, prediction, utilities)
- **catboost_info/**: CatBoost training logs and intermediate files
- **config/**: Configuration files (e.g., kaggle.json)
- **outputs/**: Model outputs and results
- **streamlit_app.py**: Streamlit web application

> Note: Scripts such as `model.py`, `train_catboost.py`, and `test_single_prediction.py` are now included in `.gitignore` and have been removed from the repository. These files are only available in your local development environment.

## Installation

### Requirements
- Python 3.8+
- To install dependencies, run:
  ```bash
  pip install -r requirements.txt
  ```
  or if you are using conda:
  ```bash
  conda env create -f environment.yml
  conda activate TelcoChurnPrediction
  ```

## Usage

1. **Train the Model**
   - Training script (if available locally):
     ```bash
     python src/train_catboost.py
     ```
2. **Run the Streamlit Application**
   - To start the web application:
     ```bash
     streamlit run streamlit_app.py
     ```