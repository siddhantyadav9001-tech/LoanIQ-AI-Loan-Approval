import numpy as np
import pandas as pd

np.random.seed(42)
N = 12000

age = np.random.randint(21, 65, N)
annual_income = np.random.randint(150000, 5000000, N)  # INR
credit_score = np.random.randint(300, 900, N)
has_ongoing_loan = np.random.randint(0, 2, N)
ongoing_loan_amount = np.where(has_ongoing_loan == 1, np.random.randint(50000, 2000000, N), 0)
has_loan_history = np.random.randint(0, 2, N)
loan_history_defaults = np.where(has_loan_history == 1, np.random.randint(0, 5, N), 0)
num_prev_loans = np.where(has_loan_history == 1, np.random.randint(1, 8, N), 0)

# Compute approval score
score = np.zeros(N)

# Credit score contribution (most important)
score += np.where(credit_score >= 750, 40, 0)
score += np.where((credit_score >= 650) & (credit_score < 750), 25, 0)
score += np.where((credit_score >= 550) & (credit_score < 650), 10, 0)
score += np.where(credit_score < 550, -20, 0)

# Income contribution
score += np.where(annual_income >= 1500000, 25, 0)
score += np.where((annual_income >= 700000) & (annual_income < 1500000), 15, 0)
score += np.where((annual_income >= 300000) & (annual_income < 700000), 5, 0)
score += np.where(annual_income < 300000, -10, 0)

# Age contribution
score += np.where((age >= 25) & (age <= 55), 10, 0)
score += np.where((age < 25) | (age > 55), -5, 0)

# Ongoing loan penalty
score += np.where(has_ongoing_loan == 0, 10, 0)
score += np.where((has_ongoing_loan == 1) & (ongoing_loan_amount > 1000000), -25, 0)
score += np.where((has_ongoing_loan == 1) & (ongoing_loan_amount <= 1000000), -10, 0)

# Loan history
score += np.where(has_loan_history == 0, 0, 0)
score += np.where((has_loan_history == 1) & (loan_history_defaults == 0), 15, 0)
score += np.where((has_loan_history == 1) & (loan_history_defaults == 1), -5, 0)
score += np.where((has_loan_history == 1) & (loan_history_defaults >= 2), -30, 0)

# Add noise
noise = np.random.normal(0, 8, N)
score += noise

# Threshold for approval
approved = (score >= 30).astype(int)

df = pd.DataFrame({
    'age': age,
    'annual_income': annual_income,
    'credit_score': credit_score,
    'has_ongoing_loan': has_ongoing_loan,
    'ongoing_loan_amount': ongoing_loan_amount,
    'has_loan_history': has_loan_history,
    'loan_history_defaults': loan_history_defaults,
    'num_prev_loans': num_prev_loans,
    'approved': approved
})

df.to_csv('loan_data.csv', index=False)
print(f"Dataset generated: {len(df)} rows")
print(f"Approval rate: {df['approved'].mean()*100:.1f}%")
print(df.head())
