import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

data = {
    "CustomerID": [f"CUST{i:05d}" for i in range(1, n+1)],
    "Tenure_Months": np.random.randint(1, 60, n),
    "Monthly_Recharge": np.round(np.random.uniform(99, 999, n), 0),
    "Recharge_Frequency": np.random.randint(1, 8, n),          # times per month
    "Avg_Data_Usage_GB": np.round(np.random.uniform(5, 150, n), 1),
    "Service_Quality_Score": np.round(np.random.uniform(1, 10, n), 1),  # 1-10
    "Call_Drop_Rate": np.round(np.random.uniform(0, 15, n), 1),
    "Complaint_Count": np.random.randint(0, 12, n),
    "Avg_Resolution_Days": np.round(np.random.uniform(0.5, 10, n), 1),
    "Competitor_Coverage_Score": np.round(np.random.uniform(1, 10, n), 1),  # higher = competitor better
    "Plan_Fit_Score": np.round(np.random.uniform(1, 10, n), 1),            # higher = better fit
    "Age": np.random.randint(18, 65, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Location_Type": np.random.choice(["Urban", "Semi-Urban", "Rural"], n),
}

df = pd.DataFrame(data)

# Realistic Churn Logic
churn_prob = (
    0.25 * (10 - df["Service_Quality_Score"]) / 10 +
    0.20 * (df["Complaint_Count"] / 12) +
    0.15 * (df["Call_Drop_Rate"] / 15) +
    0.15 * (df["Competitor_Coverage_Score"] / 10) +
    0.15 * (10 - df["Plan_Fit_Score"]) / 10 +
    0.10 * (df["Tenure_Months"] < 6).astype(int)
)

df["Churn"] = (np.random.rand(n) < churn_prob).astype(int)

df.to_csv("data/telecom_churn_data.csv", index=False)
print("Dataset created successfully with 5000 records!")
print(df["Churn"].value_counts(normalize=True))