 📊 Demographic Household Analysis and Prediction

 📌 Overview

This project performs analysis on demographic and household data using Python. It explores population distribution, household characteristics, and marital status trends, and applies a machine learning model to predict household counts based on population.

---

 🎯 Key Features

* Population aggregation and state-wise analysis
* Comparison of male and female household distribution
* Visualization of divorced population distribution
* Correlation analysis of household and marital status data
* Data normalization using Min-Max scaling
* Linear regression model for prediction

---

🛠️ Tech Stack

* Python
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

 📂 Dataset

The dataset contains demographic and household-related attributes such as:

* Total population
* Total households
* Male and female households
* Marital status categories
* Regional identifiers (state, tehsil)

---

 🔍 Workflow

 1. Data Loading

* Dataset is loaded from an Excel file
* Initial inspection of data is performed

 2. Data Exploration

* Checked for missing values
* Generated statistical summaries
* Reviewed dataset structure

 3. Data Visualization

* Pie chart for top populated states
* Bar chart for gender-based households
* Histogram for divorced persons distribution
* Heatmaps for correlation analysis

 4. Data Preprocessing

* Selected relevant numerical features
* Applied Min-Max normalization

 5. Machine Learning

* Used Linear Regression model
* Split dataset into training and testing sets
* Trained model on population vs households

 6. Evaluation

* Mean Squared Error (MSE)
* R² Score

---

 📈 Results

* Identified patterns in population distribution
* Observed gender differences in household data
* Found correlations among demographic variables
* Built a regression model for predicting household counts

---

 ▶️ How to Run

1. Install required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

2. Update dataset path in the script

3. Run the program:

```
python project.py
```

---

 ⚠️ Notes

* Ensure correct file path formatting
* Verify column names in dataset before execution
* Input values for prediction should match scaled data

---

 📌 Conclusion

This project demonstrates the application of data analysis and machine learning techniques to extract insights from demographic data and perform predictive modeling.

---

 👨‍💻 Author

Sai Ram
