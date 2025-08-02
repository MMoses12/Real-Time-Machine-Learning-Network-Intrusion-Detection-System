import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb
import joblib

# Load and clean dataset
df = pd.read_csv("Security AI/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv")
df.replace({'-': np.nan, 'http': np.nan, 'ftp': np.nan, 'tcp': np.nan}, inplace=True)
df.fillna(0, inplace=True)

# Encode proto and state
proto_encoder = LabelEncoder()
state_encoder = LabelEncoder()
df['proto'] = df['proto'].apply(lambda x: x if x in ['tcp', 'udp'] else 'other')
proto_encoder.fit(['tcp', 'udp', 'other'])
df['proto'] = proto_encoder.transform(df['proto'])

df['state'] = df['state'].apply(lambda x: x if x in ['SYN', 'FIN', '-'] else '-')
state_encoder.fit(['SYN', 'FIN', '-'])
df['state'] = state_encoder.transform(df['state'])

# Encode attack category
target = 'attack_cat'
attack_encoder = LabelEncoder()
df[target] = attack_encoder.fit_transform(df[target])

# Feature engineering
df['spkts_per_dur'] = df['spkts'] / (df['dur'] + 1e-6)
df['dpkts_per_dur'] = df['dpkts'] / (df['dur'] + 1e-6)
df['bytes_per_pkt'] = (df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts'] + 1e-6)
df['pkt_ratio'] = df['spkts'] / (df['dpkts'] + 1e-6)

# Feature selection
feature_columns = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", 
    "sload", "dload", "sinpkt", "dinpkt", "spkts_per_dur", "dpkts_per_dur",
    "proto", "state"
]
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Train/test split (NO SMOTE)
X = df[feature_columns]
y = df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y)),
    learning_rate=0.05,
    num_leaves=32,
    n_estimators=300,
    max_depth=15,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Save model (optional)
joblib.dump(model, "lightgbm_ids_model_no_smote.joblib")

# Predict and evaluate
y_pred = model.predict(X_val)
labels = attack_encoder.classes_

# 1. Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - LightGBM Multiclass Classification")
plt.tight_layout()
plt.show()

# 2. Classification Report as Bar Chart
report = classification_report(y_val, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(12, 6))
report_df.iloc[:-3][['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Precision, Recall, F1-score per Class (LightGBM)")
plt.xlabel("Class")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - LightGBM")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
