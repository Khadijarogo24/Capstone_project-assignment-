# Supermarket Sales Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)

## 📌 Project Overview

This project aims to predict the **gross income** of individual transactions in a supermarket using various machine learning models. Accurate prediction of gross income can help supermarket management in:

- Optimizing inventory based on high-profit product lines
- Staff scheduling and resource allocation
- Understanding sales patterns and customer behavior
- Strategic pricing and promotions

Two models were developed and compared:

- **Linear Regression** – a simple, interpretable baseline model
- **Random Forest Regressor** – an ensemble method capable of capturing non‑linear relationships

## 📊 Dataset

The dataset used is the **Supermarket Sales** dataset (available on [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales)). It contains 1,000 transactions recorded across three branches of a supermarket chain. Each row represents a single purchase and includes 17 features:

| Column | Description |
|--------|-------------|
| Invoice ID | Unique transaction identifier |
| Branch | Supermarket branch (A, B, C) |
| City | City of the branch |
| Customer type | Member or Normal |
| Gender | Male / Female |
| Product line | Category of the purchased product |
| Unit price | Price per unit |
| Quantity | Number of units bought |
| Tax 5% | 5% tax applied |
| Sales | Total price (including tax) |
| Date | Date of purchase |
| Time | Time of purchase |
| Payment | Payment method (Cash, Credit card, Ewallet) |
| cogs | Cost of goods sold |
| gross margin percentage | Fixed at 4.761905% |
| **gross income** | **Target variable** – profit earned |
| Rating | Customer satisfaction rating (1–10) |

**Target:** `gross income` (continuous variable)

## 🔍 Project Workflow

The analysis and modeling were performed in a Jupyter Notebook following these steps:

### 1. Exploratory Data Analysis (EDA)
- Loaded the dataset and inspected its structure (`df.info()`, `df.describe()`).
- Visualized missing values – none found.
- Correlation heatmap to identify relationships among numerical features.
- Distribution of `gross income` using a histogram with KDE.

### 2. Data Preprocessing
- Categorical variables (`Branch`, `City`, `Customer type`, `Gender`, `Product line`, `Payment`) were label‑encoded using `LabelEncoder`.
- Features (`X`) and target (`y`) were separated, with `gross income` as the target.

### 3. Train‑Test Split
- Data was split into 80% training and 20% testing sets using `train_test_split` with a fixed random state for reproducibility.

### 4. Model Building & Evaluation
Two models were trained and evaluated:

- **Linear Regression** – a baseline linear model.
- **Random Forest Regressor** – an ensemble of 100 decision trees.

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 📈 Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 5.99e-15 | 7.31e-15 | **1.0000** |
| Random Forest | 0.0376 | 0.0641 | 0.99997 |

- **Linear Regression** achieved an **almost perfect R² of 1.0**, suggesting an extremely strong linear fit. However, such perfection may indicate **data leakage** – for example, `gross income` might be directly calculable from other columns (like `Sales` and `cogs`). In practice, this model would be highly accurate if those features are available at prediction time, but caution is needed when interpreting the result.
- **Random Forest** also performed exceptionally well, with an R² of **0.99997** and very low errors. This model is more robust to non‑linearities and provides a reliable estimate even if the exact linear relationship is not present.

**Conclusion:** Both models demonstrate that gross income can be predicted with high accuracy using the available transaction features. This can be leveraged by supermarket managers for data‑driven decision‑making. The Random Forest model is recommended for deployment due to its slight advantage in capturing complex patterns and its lower risk of over‑simplification.

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Khadijarogo24/Capstone_project-assignment-.git
   cd Capstone_project-assignment-
   ```

2. **Install dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook khadija_Rogo_final_project.ipynb
   ```
   The notebook contains all code, visualizations, and model evaluations.

> **Note:** The dataset (`SuperMarket Analysis.csv`) is not included in the repository. You must download it from [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales) and place it in the same directory as the notebook, or update the file path in the notebook accordingly.

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

All required packages are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

## 📁 Repository Structure

```
├── khadija_Rogo_final_project.ipynb   # Main notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── SuperMarket Analysis.csv             # (to be downloaded separately)
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgments

- Dataset provided by [Aung Pyae](https://www.kaggle.com/aungpyaeap) on Kaggle.
- Inspired by various data science tutorials and the scikit-learn documentation.
