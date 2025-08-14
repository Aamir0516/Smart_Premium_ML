# SmartPremium: Predicting Insurance Costs with Machine Learning

This repository contains the complete workflow for SmartPremium, a machine learning project that predicts insurance premiums based on user demographic and lifestyle factors.
It includes data exploration, preprocessing, model development & evaluation, and a Streamlit web application for deployment.

🛠️ Installation & Setup

Clone the repository

git clone https://github.com/Aamir0516/SmartPremium.git
cd SmartPremium


Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows


Install dependencies

pip install -r requirements.txt

📊 Running Notebooks

For EDA:

jupyter notebook EDA.ipynb


For Preprocessing & Model Evaluation:

jupyter notebook pre&modelevaluation.ipynb

🌐 Running the Streamlit App

Make sure dependencies are installed.

Run:

streamlit run smart_premium_app.py


The app will open in your browser.

📈 Models Used

●	Linear Regression – A simple model that assumes a linear relationship between features and target.
●	Decision Trees – A model that splits the data into decision rules.
●	Random Forest – An advanced tree-based model that reduces overfitting.
●	XGBoost – A powerful gradient boosting model for high accuracy.

Evaluation metrics used:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

📌 Project Workflow

EDA — Understand the dataset through visualizations and statistical summaries.

Preprocessing — Handle missing values, encode categorical features, scale numerical data, and transform skewed distributions.

Model Development — Train multiple regression models and evaluate their performance.

Deployment — Serve predictions via Streamlit app.

👨‍💻 Author

Aamir Sohail
🔗 GitHub: Aamir0516
