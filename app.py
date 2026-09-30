from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import pickle
import os
import csv
from datetime import datetime

import os
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'loan_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

CSV_FILE = os.path.join(os.path.dirname(__file__), 'user_submissions.csv')
CSV_HEADERS = [
    'timestamp', 'name', 'age', 'annual_income', 'credit_score',
    'has_ongoing_loan', 'ongoing_loan_amount', 'ongoing_loan_type',
    'has_loan_history', 'loan_history_defaults', 'num_prev_loans',
    'prediction', 'confidence'
]

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        name = data.get('name', '').strip()
        age = int(data.get('age', 0))
        annual_income = float(data.get('annual_income', 0))
        credit_score = int(data.get('credit_score', 0))
        has_ongoing_loan = int(data.get('has_ongoing_loan', 0))
        ongoing_loan_amount = float(data.get('ongoing_loan_amount', 0)) if has_ongoing_loan else 0
        ongoing_loan_type = data.get('ongoing_loan_type', '') if has_ongoing_loan else ''
        has_loan_history = int(data.get('has_loan_history', 0))
        loan_history_defaults = int(data.get('loan_history_defaults', 0)) if has_loan_history else 0
        num_prev_loans = int(data.get('num_prev_loans', 0)) if has_loan_history else 0

        # Validate inputs
        errors = []
        if not name: errors.append("Name is required")
        if age < 18 or age > 80: errors.append("Age must be between 18 and 80")
        if annual_income <= 0: errors.append("Annual income must be positive")
        if credit_score < 300 or credit_score > 900: errors.append("Credit score must be between 300-900")

        if errors:
            return jsonify({'success': False, 'errors': errors}), 400

        # Prepare features
        features = pd.DataFrame([{
            'age': age,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'has_ongoing_loan': has_ongoing_loan,
            'ongoing_loan_amount': ongoing_loan_amount,
            'has_loan_history': has_loan_history,
            'loan_history_defaults': loan_history_defaults,
            'num_prev_loans': num_prev_loans
        }])

        # ✅ New engineered features (same as train_model.py)
        features['debt_to_income'] = features['ongoing_loan_amount'] / (features['annual_income'] + 1)
        features['credit_income_ratio'] = features['credit_score'] * features['annual_income'] / 1e9
        features['risk_score'] = features['loan_history_defaults'] * 10 - features['credit_score'] / 100

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = float(probability[prediction]) * 100

        approved = bool(prediction == 1)
        approval_prob = float(probability[1]) * 100

        # Key factors analysis
        factors = []
        if credit_score >= 750:
            factors.append({'type': 'positive', 'text': f'Excellent credit score ({credit_score})'})
        elif credit_score >= 650:
            factors.append({'type': 'neutral', 'text': f'Good credit score ({credit_score})'})
        else:
            factors.append({'type': 'negative', 'text': f'Low credit score ({credit_score}) — major concern'})

        if annual_income >= 1500000:
            factors.append({'type': 'positive', 'text': f'High annual income (₹{annual_income:,.0f})'})
        elif annual_income >= 500000:
            factors.append({'type': 'neutral', 'text': f'Moderate income (₹{annual_income:,.0f})'})
        else:
            factors.append({'type': 'negative', 'text': f'Low income (₹{annual_income:,.0f})'})

        if has_ongoing_loan:
            if ongoing_loan_amount > 1000000:
                factors.append({'type': 'negative', 'text': f'High existing debt (₹{ongoing_loan_amount:,.0f})'})
            else:
                factors.append({'type': 'neutral', 'text': f'Existing loan of ₹{ongoing_loan_amount:,.0f}'})
        else:
            factors.append({'type': 'positive', 'text': 'No ongoing loans'})

        if has_loan_history:
            if loan_history_defaults == 0:
                factors.append({'type': 'positive', 'text': 'Clean repayment history — no defaults'})
            elif loan_history_defaults == 1:
                factors.append({'type': 'neutral', 'text': f'{loan_history_defaults} past default on record'})
            else:
                factors.append({'type': 'negative', 'text': f'{loan_history_defaults} defaults in history — high risk'})

        # Save to CSV
        ensure_csv()
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                name, age, annual_income, credit_score,
                has_ongoing_loan, ongoing_loan_amount, ongoing_loan_type,
                has_loan_history, loan_history_defaults, num_prev_loans,
                'Approved' if approved else 'Rejected',
                f'{confidence:.1f}%'
            ])

        return jsonify({
            'success': True,
            'approved': approved,
            'confidence': round(confidence, 1),
            'approval_probability': round(approval_prob, 1),
            'factors': factors,
            'name': name
        })

    except Exception as e:
        return jsonify({'success': False, 'errors': [str(e)]}), 500

@app.route('/submissions')
def submissions():
    ensure_csv()
    try:
        df = pd.read_csv(CSV_FILE)
        records = df.tail(50).to_dict('records')
        return jsonify({'success': True, 'data': records, 'total': len(df)})
    except:
        return jsonify({'success': True, 'data': [], 'total': 0})

if __name__ == '__main__':
    ensure_csv()
    app.run(debug=True, port=5000)