#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import pandas as pd
import numpy as np
from datetime import timedelta
import re


# In[2]:


# Loading the dataset

clean_data = r"D:\Documents\Data Analysis\Senior Analyst Project\Main Project\Cleaned Dataset\retail_transactions.csv"
df = pd.read_csv(clean_data, low_memory=False)


# In[3]:


# Removing all transactions where customer id is "Guest"

df = df[df["customer_id"] != "Guest"]


# In[4]:


# Confirming column data types

df.dtypes


# In[5]:


# Changing the invoice date column data type to datetime:

df["invoice_date"] = pd.to_datetime(df["invoice_date"])


# In[6]:


# Confirming the data type changes 

df.dtypes


# In[7]:


# Previewing the top 5 rows of the dataset

df.head()


# In[8]:


# Converting the PostgreSQL-style booleans('t'/'f') to Python booleans (True/False)

bool_columns = ['is_return', 'is_adjustment', 'is_zero_price', 'has_suspicious_description']

for col in bool_columns:
    df[col] = df[col].map({'t': True, 'f': False})


# In[9]:


# Preparing purchase-only columns

df["purchase_quantity"] = df["quantity"].where(~df["is_return"], 0)
df["purchase_value"] = df["purchase_quantity"] * df["price"]
df["total_price"] = df["quantity"] * df["price"]  # raw, includes returns


# In[10]:


# Broad category mapping for products

broad_category_mapping = {
    "Home": [
        "Cushion", "Vase", "Mirror", "Lamp", "Light", "Lantern", "Frame", "Bunting", 
        "Garland", "Plaque", "Box", "Tin", "Holder", "Hanger", "Sign"
    ],
    "Kitchen": [
        "Mug", "Cup", "Plate", "Bowl", "Tray", "Teapot", "Coaster", "Spoon", "Jug", 
        "Napkin", "Cutlery", "Glass", "Bottle", "Canister"
    ],
    "Stationery": [
        "Pen", "Pencil", "Notebook", "Diary", "Card", "Book", "Paper", "Tape", "Tag", 
        "Envelope", "Ruler", "Sharpener", "Eraser"
    ],
    "Holiday": [
        "Christmas", "Easter", "Valentine", "Halloween", "Xmas", "Tree", "Stocking", 
        "Bauble", "Ornament"
    ],
    "Fashion": [
        "Bag", "Scarf", "Purse", "Wallet", "Hat", "Umbrella", "Mirror", "Cosmetic"
    ],
    "Toys": [
        "Toy", "Game", "Puzzle", "Ball", "Spinner", "Craft", "Kit", "Doll"
    ],
    "Garden": [
        "Plant", "Garden", "Flower", "Watering", "Pot", "Feeder", "Bird", "Butterfly"
    ],
    "Bath": [
        "Soap", "Bath", "Towel", "Sponge", "Lotion", "Wash", "Gel"
    ],
    "Jewelry": [
        "Necklace", "Bracelet", "Ring", "Charm", "Jewellery"
    ],
    "Others": [
        "Postage", "Unknown", "Adjustment", "Error", "Misc"
    ]
}

# Invert mapping for quick lookup
keyword_to_category = {
    keyword: category
    for category, keywords in broad_category_mapping.items()
    for keyword in keywords
}

def map_to_category(description):
    if pd.isna(description) or description == "Unknown":
        return "Others"
    words = re.findall(r'\b[A-Z][a-z]+\b', description)  # Title Case words
    for word in words:
        if word in keyword_to_category:
            return keyword_to_category[word]
    return "Others"

df["broad_category"] = df["description"].apply(map_to_category)


# In[11]:


# Churn labelling (Shift-based)

df = df.sort_values(["customer_id", "invoice_date"])
df["next_purchase_date"] = df.groupby("customer_id")["invoice_date"].shift(-1)
df["days_to_next"] = (df["next_purchase_date"] - df["invoice_date"]).dt.days
label_window = 90
df["churn_label"] = ((df["days_to_next"].isna()) | (df["days_to_next"] > label_window)).astype(int)


# In[12]:


# Precomputing return quantity for ratios

df["return_qty"] = np.where(df["is_return"], df["quantity"].abs(), 0)


# In[13]:


# Tenure and purchase intervals

first_purchase = df.groupby("customer_id")["invoice_date"].transform("min")
df["tenure_days"] = (df["invoice_date"] - first_purchase).dt.days
df["prev_purchase_date"] = df.groupby("customer_id")["invoice_date"].shift(1)
df["days_since_prev"] = (df["invoice_date"] - df["prev_purchase_date"]).dt.days
interval_stats = (
    df.groupby("customer_id")["days_since_prev"]
      .agg(["mean", "std"])
      .reset_index()
      .rename(columns={"mean": "purchase_interval_mean", "std": "purchase_interval_std"})
)


# In[14]:


# Feature engineering 

window_days = 90

def compute_features(group):
    g = group.sort_values("invoice_date").copy()
    g = g.set_index("invoice_date")

    # Count distinct invoices in the last 90 days
    g["is_new_invoice"] = (g["invoice"] != g["invoice"].shift()).astype(int)
    rolling_90 = g.rolling(f"{window_days}D", closed="left")
    orders_count_90 = rolling_90["is_new_invoice"].sum()

    # Purchase-based rolling sums
    qty_sum_90 = rolling_90["purchase_quantity"].sum()
    spend_sum_90 = rolling_90["purchase_value"].sum()

    # Return metrics
    return_ratio_90 = rolling_90["is_return"].mean()
    return_qty_90 = rolling_90["return_qty"].sum()
    return_qty_ratio_90 = return_qty_90 / (qty_sum_90 + 1e-9)

    # Category diversity & top category share
    cat_qty = (
        g.pivot_table(
            index=g.index, columns="broad_category",
            values="purchase_quantity", aggfunc="sum"
        )
        .reindex(g.index)  # match main index
        .fillna(0.0)
    )
    cat_qty_roll90 = cat_qty.rolling(f"{window_days}D", closed="left").sum()
    category_diversity_90d = (cat_qty_roll90 > 0).sum(axis=1).astype(float)
    cat_total = cat_qty_roll90.sum(axis=1)
    cat_max = cat_qty_roll90.max(axis=1)
    top_category_share_90d = (cat_max / cat_total.replace(0, np.nan)).fillna(0.0)

    # Averages & trends
    avg_items_per_order_90d = (qty_sum_90 / orders_count_90.replace(0, np.nan)).fillna(0.0)
    avg_order_value_90d = (spend_sum_90 / orders_count_90.replace(0, np.nan)).fillna(0.0)
    spend_trend = (spend_sum_90 - spend_sum_90.shift(1)) / (spend_sum_90.shift(1) + 1e-9)
    order_trend = (orders_count_90 - orders_count_90.shift(1)) / (orders_count_90.shift(1) + 1e-9)

    # ✅ Force alignment to g.index for every Series
    def align(s):
        return s.reindex(g.index).values

    out = pd.DataFrame({
        "customer_id": g["customer_id"].values,
        "invoice_date": g.index,
        "category_diversity_90d": align(category_diversity_90d),
        "top_category_share_90d": align(top_category_share_90d),
        "avg_items_per_order_90d": align(avg_items_per_order_90d),
        "avg_order_value_90d": align(avg_order_value_90d),
        "spend_trend": align(spend_trend),
        "order_trend": align(order_trend),
        "return_ratio_90d": align(return_ratio_90),
        "return_qty_ratio_90d": align(return_qty_ratio_90)
    })
    return out.reset_index(drop=True)


# In[15]:


# Merging features + labels 

labels_df = (
    df[["customer_id", "invoice_date", "churn_label"]]
    .drop_duplicates(subset=["customer_id", "invoice_date"])
)

features_df = (
    df[["customer_id", "invoice_date", "purchase_quantity", "purchase_value",
        "is_return", "return_qty", "broad_category", "quantity", "price", "invoice"]]
    .groupby("customer_id", group_keys=False)
    .apply(compute_features)
)

feature_df = (
    features_df
    .merge(labels_df, on=["customer_id", "invoice_date"], how="left")
    .merge(interval_stats, on="customer_id", how="left")
)

feature_df.head()


# In[16]:


# Validating and Finalising
# Checking for NaNs / Infs
feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(feature_df.isna().sum())

# Deciding how to handle NaNs
feature_df.fillna(0, inplace=True)  # or impute/drop depending on context

# Downcasting numerics to save memory
for col in feature_df.select_dtypes(include=["float64", "int64"]).columns:
    feature_df[col] = pd.to_numeric(feature_df[col], downcast="float" if "float" in str(feature_df[col].dtype) else "integer")

# Confirming churn_label is clean
assert feature_df["churn_label"].notna().all(), "Missing churn labels detected!"


# In[17]:


# Training and Evaluating Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

X = feature_df.drop(columns=["customer_id", "invoice_date", "churn_label"])
y = feature_df["churn_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Baseline: Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
print("LogReg ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

# Boosted: LightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred_proba_lgbm = lgbm.predict_proba(X_test)[:, 1]
print("LightGBM ROC-AUC:", roc_auc_score(y_test, y_pred_proba_lgbm))


# In[18]:


# Calibrating and interpreting
import shap

# Taking a random sample of rows for speed
sample_size = 1000
sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X.iloc[sample_idx]

# Creating the explainer
explainer = shap.TreeExplainer(lgbm)

# Computing SHAP values on the sample
shap_values = explainer.shap_values(X_sample)

# Summary plot
shap.summary_plot(shap_values, X_sample)


# In[19]:


# Adding churn risk scores

feature_df["churn_risk_score"] = lgbm.predict_proba(X)[:, 1]
feature_df["at_risk"] = (feature_df["churn_risk_score"] >= 0.6).astype(int)


# In[20]:


# Handling class imbalance explicitly

lgbm = lgb.LGBMClassifier(
    random_state=42,
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
)


# In[21]:


# Retraining the model with imbalance handling
# To retrain with class imbalance weight
lgbm.fit(X_train, y_train)

# Evaluate again
y_pred_proba = lgbm.predict_proba(X_test)[:, 1]
print("LightGBM ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, (y_pred_proba >= 0.6).astype(int)))


# In[22]:


# Tuning the probability threshold instead of going all in on scale_pos_weight
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# For instance: picking a threshold where recall ~0.5 and precision is acceptable
target_recall = 0.5
idx = np.argmin(np.abs(recalls - target_recall))
best_threshold = thresholds[idx]

print(f"Chosen threshold: {best_threshold:.2f}")
print(f"Precision: {precisions[idx]:.2f}, Recall: {recalls[idx]:.2f}")

# Applying to feature_df
feature_df["at_risk"] = (feature_df["churn_risk_score"] >= best_threshold).astype(int)


# In[23]:


# Lowering the threshold to balance precision and recall
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Example: pick threshold where precision >= 0.2 and recall >= 0.5
mask = (precisions >= 0.2) & (recalls >= 0.5)
best_idx = np.argmax(mask)  # first threshold that meets criteria
chosen_threshold = thresholds[best_idx]

print(f"Chosen threshold: {chosen_threshold:.2f}")
print(f"Precision: {precisions[best_idx]:.2f}, Recall: {recalls[best_idx]:.2f}")


# In[24]:


# Retraining LightGBM with softer imbalance handling
lgbm = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31
)
lgbm.fit(X_train, y_train)

y_pred_proba = lgbm.predict_proba(X_test)[:, 1]
print("LightGBM ROC-AUC:", roc_auc_score(y_test, y_pred_proba))


# In[25]:


# Tuning the threshold for balance
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# Example: pick threshold where recall ~0.5
target_recall = 0.5
best_idx = (np.abs(recalls - target_recall)).argmin()
chosen_threshold = thresholds[best_idx]
print(f"Chosen threshold: {chosen_threshold:.2f}")
print(f"Precision: {precisions[best_idx]:.2f}, Recall: {recalls[best_idx]:.2f}")


# In[26]:


# Applying to the dataset
feature_df["churn_risk_score"] = lgbm.predict_proba(X)[:, 1]
feature_df["at_risk"] = (feature_df["churn_risk_score"] >= chosen_threshold).astype(int)
print(feature_df["at_risk"].value_counts(normalize=True))


# In[27]:


# --- 1️⃣ Get probabilities from your trained model ---
# Make sure y_pred_proba is defined from your latest model:
# y_pred_proba = lgbm.predict_proba(X_test)[:, 1]

# --- 2️⃣ Scan thresholds ---
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Put into DataFrame for easy filtering
pr_df = pd.DataFrame({
    "threshold": thresholds,
    "precision": precisions[:-1],  # last precision/recall pair has no threshold
    "recall": recalls[:-1]
})

# --- 3️⃣ Filter for business-viable candidates ---
# Adjust these floors to your needs
min_precision = 0.20
min_recall = 0.40

candidates = pr_df[(pr_df["precision"] >= min_precision) & (pr_df["recall"] >= min_recall)]

if not candidates.empty:
    # Compute F1 score for each candidate
    candidates["f1"] = 2 * (candidates["precision"] * candidates["recall"]) / (candidates["precision"] + candidates["recall"])
    # Pick the candidate with the highest F1
    best_row = candidates.sort_values("f1", ascending=False).iloc[0]
    chosen_threshold = best_row["threshold"]
    print(f"Chosen threshold: {chosen_threshold:.2f}")
    print(f"Precision: {best_row['precision']:.2f}, Recall: {best_row['recall']:.2f}, F1: {best_row['f1']:.2f}")
else:
    # Fallback if no threshold meets criteria
    chosen_threshold = 0.5
    print(f"No threshold met the criteria — using default {chosen_threshold}")

# --- 4️⃣ Apply to full dataset ---
# Make sure feature_df["churn_risk_score"] already exists or create it:
# feature_df["churn_risk_score"] = lgbm.predict_proba(X)[:, 1]

feature_df["at_risk"] = (feature_df["churn_risk_score"] >= chosen_threshold).astype(int)

# --- 5️⃣ Inspect proportion flagged ---
print("Proportion flagged as at risk (full dataset):")
print(feature_df["at_risk"].value_counts(normalize=True).rename("proportion"))


# In[28]:


# Dropping some of the columns that are causing noise for the model
# The dropped columns had thousands of missing values that were filled with 0s which is affecting the model performance
drop_cols = [
    "spend_trend",
    "order_trend",
    "return_ratio_90d",
    "return_qty_ratio_90d"
]
feature_df = feature_df.drop(columns=drop_cols)


# In[29]:


# Retraining the model on the leaner feature set
# Define features after dropping
drop_cols = ["spend_trend", "order_trend", "return_ratio_90d", "return_qty_ratio_90d"]

# This will drop them if they exist, skip silently if they don't
feature_df = feature_df.drop(columns=drop_cols, errors="ignore")

X = feature_df.drop(columns=["customer_id", "invoice_date", "churn_label"])
y = feature_df["churn_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

lgbm = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31
)
lgbm.fit(X_train, y_train)

y_pred_proba = lgbm.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, (y_pred_proba >= 0.5).astype(int)))


# In[30]:


# Choosing a practical cut off for the app using top-k approach
feature_df["churn_risk_score"] = lgbm.predict_proba(X)[:, 1]

top_k_rate = 0.05  # top 5% highest risk
chosen_threshold = feature_df["churn_risk_score"].quantile(1 - top_k_rate)

feature_df["at_risk"] = (feature_df["churn_risk_score"] >= chosen_threshold).astype(int)

print(f"Chosen threshold: {chosen_threshold:.4f}")
print(feature_df["at_risk"].value_counts(normalize=True).rename("proportion"))


# In[31]:


# Exporting the Model for Deployment (Using joblib)
import joblib

# Saving the trained model
joblib.dump(lgbm, "new_churn_prediction_model.pkl")


# In[34]:


# Creating a demo dataset for the app testing
from datetime import datetime, timedelta
import random
import os

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Broad category mapping from training
broad_category_mapping = {
    "Home": [
        "Cushion", "Vase", "Mirror", "Lamp", "Light", "Lantern", "Frame", "Bunting", 
        "Garland", "Plaque", "Box", "Tin", "Holder", "Hanger", "Sign"
    ],
    "Kitchen": [
        "Mug", "Cup", "Plate", "Bowl", "Tray", "Teapot", "Coaster", "Spoon", "Jug", 
        "Napkin", "Cutlery", "Glass", "Bottle", "Canister"
    ],
    "Stationery": [
        "Pen", "Pencil", "Notebook", "Diary", "Card", "Book", "Paper", "Tape", "Tag", 
        "Envelope", "Ruler", "Sharpener", "Eraser"
    ],
    "Holiday": [
        "Christmas", "Easter", "Valentine", "Halloween", "Xmas", "Tree", "Stocking", 
        "Bauble", "Ornament"
    ],
    "Fashion": [
        "Bag", "Scarf", "Purse", "Wallet", "Hat", "Umbrella", "Mirror", "Cosmetic"
    ],
    "Toys": [
        "Toy", "Game", "Puzzle", "Ball", "Spinner", "Craft", "Kit", "Doll"
    ],
    "Garden": [
        "Plant", "Garden", "Flower", "Watering", "Pot", "Feeder", "Bird", "Butterfly"
    ],
    "Bath": [
        "Soap", "Bath", "Towel", "Sponge", "Lotion", "Wash", "Gel"
    ],
    "Jewelry": [
        "Necklace", "Bracelet", "Ring", "Charm", "Jewellery"
    ],
    "Others": [
        "Postage", "Unknown", "Adjustment", "Error", "Misc"
    ]
}

# Flatten all keywords into a list of valid descriptions
all_descriptions = [item for sublist in broad_category_mapping.values() for item in sublist]

# Parameters
num_customers = 40
num_rows = 300
min_tx_per_customer = 2

# Generate customer IDs
customer_ids = [f"CUST{str(i).zfill(3)}" for i in range(1, num_customers + 1)]

# Ensure at least 2 transactions per customer
base_data = []
for cust_id in customer_ids:
    for _ in range(min_tx_per_customer):
        base_data.append(cust_id)

# Fill remaining rows randomly
remaining = num_rows - len(base_data)
base_data += list(np.random.choice(customer_ids, remaining))
np.random.shuffle(base_data)

# Date range
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 8, 31)
days_range = (end_date - start_date).days

# Generate dataset
data = []
for cust_id in base_data:
    invoice_date = start_date + timedelta(days=random.randint(0, days_range))
    invoice = f"INV{random.randint(10000, 99999)}"
    quantity = random.randint(1, 10)  # positive only
    price = round(random.uniform(5.0, 100.0), 2)
    description = random.choice(all_descriptions)
    data.append([cust_id, invoice_date.strftime("%Y-%m-%d"), invoice, quantity, price, description])

# Create DataFrame
df_demo = pd.DataFrame(data, columns=[
    "customer_id", "invoice_date", "invoice", "quantity", "price", "description"
])

# Save to Downloads
save_path = r"C:\Users\admin\Downloads\simulated_churn_dataset.csv"
df_demo.to_csv(save_path, index=False)

print(f"Dataset saved to: {save_path}")
print(df_demo.head())


# In[ ]:





# In[ ]:




