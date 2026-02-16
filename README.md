# LendingClub Analytics & Prediction

A comprehensive tool for exploring LendingClub historical data and predicting loan approval odds using state-of-the-art Machine Learning.

## 🚀 Projects Overview

This repository now contains two versions of the application:
1.  **Modern Full-Stack Version**: A Next.js + FastAPI application with LightGBM.
2.  **Legacy Dash Version**: The original Dash/Plotly application with recent visual polish.

---

## ✨ Modern Version (2024 Rewrite)

The modern version is a full-stack rewrite designed for high performance and premium UI/UX.

### Architecture
- **Backend**: FastAPI, Poetry, LightGBM (80.25% accuracy)
- **Frontend**: Next.js 16, TypeScript, Tailwind CSS v4, Shadcn/UI
- **Features**: 
  - **Market Insights**: Interactive Recharts dashboard for 2007-2017 loan data.
  - **AI Pre-Approval**: Real-time logic-aware risk assessment.
  - **Premium UI**: Glassmorphism, animated transitions, and responsive design.

### Running the Modern Version
1. **Start Backend**:
   ```bash
   cd lendingclub-modern/backend
   poetry run uvicorn main:app --reload --port 8000
   ```
2. **Start Frontend**:
   ```bash
   cd lendingclub-modern/frontend
   npm run dev
   ```
   Access at: [http://localhost:3000](http://localhost:3000)

---

## 📊 Legacy Version (Original Dash App)

The original interactive tool built with Dash and Plotly for data exploration.

### Features
- **Investor EDA**: Global views of loan status and regional distributions.
- **Predictor**: Logistic Regression and Random Forest models for odds estimation.

### Running the Legacy Version
```bash
python index.py
```
Access at: [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 📂 Dataset Info
The project utilizes the LendingClub dataset (2007-2017) originally from Kaggle.
- **Size**: ~700MB raw data.
- **Cleaning**: Pre-processed in Jupyter to handle feature selection and aggregation.

## 🛠️ Requirements
- **Python 3.11+**
- **Node.js 25+**
- **Poetry** (for modern backend)
