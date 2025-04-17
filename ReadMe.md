# 🚀 AutoVizor: Your AI-Powered Data Exploration & Modeling Assistant

AutoVizor is an all-in-one Streamlit-based application that helps you **upload**, **analyze**, **clean**, and **model** your datasets in seconds. With built-in EDA, outlier detection, AutoML, model interpretability, and time-series forecasting support, AutoVizor is your ultimate no-code data companion. 📊💡

---

## ✨ Features

### 📂 File Upload
- Accepts `.csv` and `.xlsx` files.
- Displays a preview and schema of your dataset.

### 🔍 Automated EDA
- Summary stats (`.describe()`, `.info()`)
- Missing value heatmaps (via `missingno`)
- Distributions for numerical and categorical features
- Correlation matrix
- Full profiling report using `ydata-profiling`

### 🧹 Data Cleaning
- Drop missing values
- Fill missing values (mean/median/mode)
- Export cleaned dataset as `.csv`

### 🚨 Outlier Detection
- Z-score based detection
- Visualize outliers in numerical columns

### 🤖 AutoML Modeling
- Train classification/regression models using `PyCaret`
- Compare multiple models instantly
- Choose target column dynamically

### 🧠 Model Interpretability
- SHAP plots for feature importance
- Global and local model explanations

### ⏳ Time-Series Forecasting
- Automatic time-series modeling using `Prophet`
- Custom date and value columns selection
- Forecast visualizations

### 💾 Export Options
- Download cleaned dataset
- Export trained models (coming soon!)

---

## 🛠 Installation

### ✅ Prerequisites

- Python 3.8 or higher
- pip

### 📦 Install Dependencies

```bash
git clone https://github.com/yourusername/autovizor.git
cd autovizor
pip install -r requirements.txt
