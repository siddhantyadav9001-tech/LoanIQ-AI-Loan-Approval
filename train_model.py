import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle

df = pd.read_csv('loan_data.csv')

# ✅ NEW: Feature Engineering — naye features banao
df['debt_to_income'] = df['ongoing_loan_amount'] / (df['annual_income'] + 1)
df['credit_income_ratio'] = df['credit_score'] * df['annual_income'] / 1e9
df['risk_score'] = df['loan_history_defaults'] * 10 - df['credit_score'] / 100

features = [
    'age', 'annual_income', 'credit_score',
    'has_ongoing_loan', 'ongoing_loan_amount',
    'has_loan_history', 'loan_history_defaults', 'num_prev_loans',
    'debt_to_income', 'credit_income_ratio', 'risk_score'  # NEW
]

X = df[features]
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ NEW: Ensemble — 3 models milake predict karo
rf = RandomForestClassifier(
    n_estimators=500,       # 200 → 500
    max_depth=15,           # 12 → 15
    min_samples_split=3,    # 5 → 3
    min_samples_leaf=1,     # 2 → 1
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Voting Classifier — dono milke decide karte hain
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ensemble)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

with open('loan_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved!")