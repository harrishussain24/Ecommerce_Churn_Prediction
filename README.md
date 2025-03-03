Customer Churn Prediction - Machine Learning Project

Overview
This project aims to predict customer churn using machine learning teechniques. We analyze factors such as monthly spending, subscription length, contract type, and premium support to understand why customers leave.

Dataset
The dataset used in this project is customer_churn_synthetic.csv. It contains various attributes related to cutomer behavior, including:
    - Subscription Length (Months)
    - Monthly Spending (Amount in $)
    - Total Spent (Total Amount Spent)
    - Contract Type (Monthly or Yearly)
    - Premium Support (Yes or No)
    - Churn (0 = Active, 1 = Churned)

Dependencies
Ensure you have the following Python libraries installed: pip install pandas matplotlib seaborn scikit-learn scipy joblib

Exploatory Data Analysis
* Churn Distribution
    * Identified that 26% of customers churned.
    * Monthly subscribers churn more than yearly subscribers.
* Feature Analysis 
    * Higher Monthly Spending -> More Churn
    * Longer Subscription -> Less Churn
    * Premium Support -> Lower Churn Rate

Visualizations 
The project uses matplotlib and seaborn to generate:
* churn_status vs monthly_spending
* churn_status vs subscripiton_length
* contract_type vs number_of_customers
* number_of_customers vs churn_distribution
* premium_support vs number_of_customers
* importance_score vs feature

Data Preprocessing
* Converted categorical features (ContractType, HasPremiumSupport) into numeric form.
* Scaled numerical features (SubscriptionLength, MonthlySpending, TotalSpent) using Z-score normalization (Scipy).
* Split dataset into 80% training and 20% testing.

Machine Learning Models
We trained and evaluated two models:
* Logistic Regression 
		Accuracy -> 74%, Precision -> 67%, Recall -> 8%, F1_Score -> 14%
* Random Forest
		Accuracy -> 74%, Precision -> 52%, Recall -> 25%, F1_Score -> 33%

Hyperparameter Tuning
We used GridSearchCV to find the best parameters for RandomForestClassifier and the best parameters we found are:
* n_estimators: 50
* max_depth: 10
* min_samples_leaf: 4
* min_samples_split: 10 
 Accuracy improved from 74% to 76%.

Model Saving and Deployment
We saved the model using Joblib and loaded it for real-time predictions.

How to Run the Project 
* Setting Up the Project
    * Go to the GitHub repository and download the project as a ZIP file and extract it to your desired location.
    * Alternatively you can clone the repository using Git.
* Run Data Analysis & Preprocessing
    * Open exploatory_data_analysis.ipynb in Jupyter Notebook.
    * Run the notebook cells to perform data cleaning, visualization, and preprocessing.
* 3. Train the Model
    * Open training_model.ipynb in Jupyter Notebook.
    * Run all cells to train the model using processed data.
    * The trained model will be saved as random_forest_churn_model.pkl.
* Use the Trained Model for Predictions
    * Create a new file predict.py (or a Jupyter Notebook like model_prediction.ipynb).
    * Load the trained model and make predictions on new data.
    * Example usage:
        * import joblib 
        * import pandas as pd 
        * Load trained model  -> model = joblib.load("trained_model.pkl") 
        * Load new data  -> new_data = pd.read_csv("new_data.csv") 
        * Make predictions  -> predictions = model.predict(new_data) 
        * print("Predictions:", predictions)
* Additional Notes
    * If you encounter errors, ensure all dependencies in requirements.txt are installed.
    * If you use Jupyter Notebook, execute cells sequentially to avoid errors.
    * The outputs will include printed insights, model metrics, and visualizations.

Future Improvements 
* Enhance Model Performance by fine-tuning and hyperparameters and testing advanced models like XGBoost or LightGBM
* Improve Class Imbalance Handling using techniques like SMOTE or class-weight adjustments.
* Expand Feature Engineering by incorporating customer behavior trends, interaction frequency and spending patterns.
* Develop an Interactive Dashboard to visualize churn predictions and key insights.
* Automate Predictions by deploying the model as a web app using Flask or FastAPI.

Author 
Harris Hussain harrishussain2408@gmail.com

License
This project is open-source under MIT License.
