# 🏦 LoanIQ — AI Loan Approval System

> AI-powered loan eligibility predictor using Random Forest ML model with 92.4% accuracy.

---

## 📁 Project Structure

```
loan_approval/
├── templates/
│   └── index.html         → Interactive frontend (dark theme UI)
├── app.py                 → Flask backend + REST API
├── generate_data.py       → 12,000-row synthetic dataset generator
├── train_model.py         → Random Forest ML model trainer
├── loan_data.csv          → Generated training data (12,000 rows)
├── loan_model.pkl         → Trained ML model (auto-generated)
└── user_submissions.csv   → User prediction log (auto-created)
```

---

## ⚙️ Setup & Run

### Step 1: Install dependencies
```bash
pip install flask scikit-learn pandas numpy
```

### Step 2: Generate training data
```bash
python generate_data.py
```

### Step 3: Train the ML model
```bash
python train_model.py
```

### Step 4: Start Flask server
```bash
python app.py
```

### Step 5: Open in browser
```
http://localhost:5000
```

---

## 🤖 ML Model Details

| Parameter     | Value                     |
|---------------|---------------------------|
| Algorithm     | Random Forest (200 trees) |
| Training Rows | 12,000                    |
| Accuracy      | **92.4%**                 |
| ROC-AUC       | **0.978**                 |

### Feature Importance (in order):
1. **Credit Score** — 57%
2. **Loan History Defaults** — 11%
3. **Ongoing Loan Amount** — 9%
4. **Annual Income** — 8%
5. **Age** — 6%
6. **Has Ongoing Loan** — 5%
7. **Number of Previous Loans** — 2%
8. **Has Loan History** — 1%

---

## 🌐 API Endpoints

### POST `/predict`

**Request body:**
```json
{
  "name": "Rahul Sharma",
  "age": 30,
  "annual_income": 800000,
  "credit_score": 720,
  "has_ongoing_loan": 0,
  "ongoing_loan_amount": 0,
  "ongoing_loan_type": "",
  "has_loan_history": 1,
  "loan_history_defaults": 0,
  "num_prev_loans": 2
}
```

**Response:**
```json
{
  "success": true,
  "approved": true,
  "confidence": 97.1,
  "approval_probability": 97.1,
  "factors": [
    { "type": "positive", "text": "Good credit score (720)" },
    { "type": "positive", "text": "No ongoing loans" }
  ],
  "name": "Rahul Sharma"
}
```

### GET `/submissions`
Returns last 50 user submissions from CSV log.

---

## 📊 Approval Scoring Logic

| Factor               | Condition              | Points |
|----------------------|------------------------|--------|
| Credit Score         | 750+                   | +40    |
| Credit Score         | 650–749                | +25    |
| Credit Score         | 550–649                | +10    |
| Credit Score         | Below 550              | -20    |
| Annual Income        | ₹15L+                  | +25    |
| Annual Income        | ₹7L–15L                | +15    |
| Annual Income        | ₹3L–7L                 | +5     |
| Annual Income        | Below ₹3L              | -10    |
| Age                  | 25–55                  | +10    |
| Age                  | Outside 25–55          | -5     |
| No Ongoing Loan      | —                      | +10    |
| Ongoing Loan > ₹10L  | —                      | -25    |
| Ongoing Loan ≤ ₹10L  | —                      | -10    |
| Clean History        | 0 defaults             | +15    |
| 1 Default            | —                      | -5     |
| 2+ Defaults          | —                      | -30    |

> **Threshold:** Score ≥ 30 → Approved

---

## 📝 Notes

- `user_submissions.csv` is auto-created on first prediction
- `loan_model.pkl` must exist before running `app.py` (run `train_model.py` first)
- All monetary values are in Indian Rupees (INR)
