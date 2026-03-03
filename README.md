Problem Statement (Theme : Predict the Fraud)

Current fraud detection systems can only identify known fraud patterns.
Attackers are changing and adapting much quicker than current rule-based systems. This causes fraud to occur outside of current detection systems.

Goal:

Develop a fraud intelligence system to identify fraudulent intent from behavioural context before fraud patterns are established. This should include low false-positive rates.

💡 Solution Overview

FraudSense is an AI-based fraud detection system. This system will analyse transactional behavior and user activity patterns to predict fraud.

Instead of checking against known rules (like checking if the amount is too large), we:

Will create behavioural profiles for users.
Will identify anomalies in transaction patterns.
Will calculate a risk score.
Will flag suspicious transactions before fraud is actually committed.

Anomaly detection and risk scoring will identify suspicious intent.

🏗️ System Architecture

User → Web Application → API Server → Feature Engineering → Fraud Detection Model → Risk Scoring Engine → Database → Alert System

🛠️ Planned Tech Stack

Frontend: HTML + React
Backend: Flask + Node.js
Database: SQLite + MongoDB
ML Model: Isolation Forest + Logistic Regression




