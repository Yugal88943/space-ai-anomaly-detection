# Space AI Project

## AI-Based Real-Time Anomaly Detection and Autonomous Decision Support System for Deep Space Missions

---

## Installation & Setup

```bash
# Clone repository
git clone <>
cd space_ai_project

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard/app.py
```

---

## Technology Stack

### 🔹 Programming Language

* Python

---

### 🔹 Libraries

#### Data Handling

* pandas
* numpy

#### Visualization

* matplotlib
* seaborn

#### Machine Learning

* scikit-learn
* xgboost

#### Dashboard

* streamlit

---

### 🔹 Tools

* Jupyter Notebook
* VS Code
* GitHub

---

## System Requirements

### Minimum

* RAM: 8 GB
* CPU: Intel i5 / Ryzen 5
* Storage: 5–10 GB

### Recommended

* RAM: 16 GB
* GPU: Optional

---

## Results

* RMSE: ~18
* MAE: ~13
* R² Score: ~0.80
* Anomaly Detection F1-score: ~0.81
* High recall (~0.93) ensures reliable anomaly detection

---

## Features

* Real-time anomaly detection
* Remaining Useful Life (RUL) prediction
* Risk scoring system
* Autonomous decision engine
* Interactive dashboard visualization

---

## Future Work

* LSTM-based time-series modeling
* Autoencoder-based anomaly detection
* Integration with real-time telemetry data

---


space_ai_project/
│
├── dashboard/
│   ├── app.py
|
├── data/
│   ├── raw/
│   	├──train_FD001.txt
│   	├──test_FD001.txt
│   	├──RUL_FD001.txt	
│   	
│
├── notebooks/
│   ├──Project.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── anomaly_detection.py
│   ├── failure_prediction.py
│   ├── decision_engine.py
│   ├── run_pipeline.py
│   
│
├── report/
│   ├──Presentation
|       ├──AI_Space.pdf
|     ── report.docx  
│
├── requirements.txt
├── README.md