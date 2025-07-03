# Telco Customer Churn Prediction

This project predicts customer churn for a telecommunications company using machine learning techniques. The solution is built with Python and leverages the CatBoost algorithm for classification, with a Streamlit web app for interactive predictions.

## Features
- Data preprocessing and feature engineering
- Model training with CatBoost
- Model evaluation and visualization
- Streamlit app for user-friendly predictions
- Modular code structure for easy maintenance

## Project Structure
```
├── README.md
├── requirements.txt
├── streamlit_app.py
├── catboost_info/
├── config/
├── data/
├── models/
├── notebooks/
├── outputs/
└── src/
```
- **data/**: Raw dataset(s)
- **models/**: Trained models and pipelines
- **src/**: Source code (data processing, training, prediction, visualization)
- **notebooks/**: Jupyter notebooks for exploration and feature engineering
- **streamlit_app.py**: Streamlit web application

## Getting Started

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Usage
1. **Train the Model**
   - Run the training script:
     ```bash
     python src/train_catboost.py
     ```
2. **Run the Streamlit App**
   - Start the web app:
     ```bash
     streamlit run streamlit_app.py
     ```

## Data
The dataset is located in the `data/` folder: `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Model
The trained CatBoost model and pipelines are saved in the `models/` directory.

## Authors
- [Your Name]

## License
This project is licensed under the MIT License.

