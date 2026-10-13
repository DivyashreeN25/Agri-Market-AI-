# Best Market Selection and Profit Prediction System for Farmers

An intelligent **decision support system** that helps farmers choose the most profitable market for selling crops. The system uses **Machine Learning (XGBoost)** to predict future crop prices, calculates expected profits, ranks markets, analyzes risk, and recommends the optimal market.

---

## Features

| Module | Description |
|--------|-------------|
| **Farmer Module** | Registration, login, profile management |
| **Crop Module** | Add crops with quantity, production cost, category |
| **Market Module** | Manage markets with distance, transport cost, demand/supply |
| **ML Module** | XGBoost price prediction with preprocessing pipeline |
| **Profit Module** | Revenue and profit calculation per market |
| **Recommendation** | Composite scoring: 60% profit + 25% stability + 15% distance |
| **Risk Analysis** | Price volatility, demand/supply fluctuation scoring |
| **Dashboard** | Professional Streamlit UI with charts and KPIs |
| **Reports** | CSV export for profit and market ranking |

---

## Tech Stack

- Python 3.12
- Streamlit
- MySQL + SQLAlchemy ORM
- Pandas, NumPy
- Scikit-Learn, XGBoost
- Plotly, Matplotlib
- Pickle (model persistence)

---

## Project Structure

```
project/
├── app.py                      # Streamlit dashboard
├── config.py                   # Configuration
├── setup.py                    # Data + ML + DB setup
├── requirements.txt
├── database/
│   ├── connection.py           # SQLAlchemy engine
│   └── schema.sql              # MySQL schema
├── models/
│   ├── farmer.py               # Farmer CRUD
│   ├── crop.py                 # Crop CRUD
│   ├── market.py               # Market CRUD
│   └── prediction.py           # Predictions & recommendations
├── ml/
│   ├── preprocessing.py        # Cleaning, encoding, features
│   ├── train_model.py          # XGBoost training
│   ├── predict.py              # Inference
│   └── model.pkl               # Saved model (after training)
├── services/
│   ├── profit_service.py
│   ├── risk_service.py
│   └── recommendation_service.py
├── data/
│   └── (CSV files deprecated - using live market data)
├── reports/                    # Generated CSV reports
└── utils/
    ├── helpers.py
    └── generate_dataset.py
```

---

## Installation

### 1. Prerequisites

- Python 3.12+
- MySQL Server 8.0+

### 2. Clone and Setup Virtual Environment

```bash
cd "d:\Market price prediction"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Database (Optional)

Copy `.env.example` to `.env` and set credentials:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=farmer_market_db
```

Create database:

```bash
mysql -u root -p < database/schema.sql
```

### 4. Run Setup (Generate Data + Train Model + Seed DB)

```bash
python setup.py
```

### 5. Launch Dashboard

```bash
streamlit run app.py
```

Open: **http://localhost:8501**

**Note:** MySQL is required. The application uses live market data from Data.gov.in API.

---

## Algorithm Explanation

### 1. Price Prediction (XGBoost)

**Input Features:**
- Crop Name, Market Name (label encoded)
- Month (with cyclical sin/cos encoding)
- Historical Price, Demand, Supply
- Transportation Cost, Distance
- Engineered: demand/supply ratio, market pressure, price per km

**Output:** Predicted future crop price

**Metrics:** MAE, RMSE, R² on 80/20 train-test split

### 2. Profit Calculation

```
Revenue = Predicted Price × Quantity
Profit = Revenue − Production Cost − Transportation Cost
```

### 3. Risk Analysis

Risk score from:
- **Price volatility** (50% weight)
- **Demand fluctuation** (25% weight)
- **Supply fluctuation** (25% weight)

Categories: Low Risk (<30), Medium Risk (30-60), High Risk (>60)

### 4. Final Recommendation Score

```
Final Score = (0.60 × Profit Score) + (0.25 × Stability Score) + (0.15 × Distance Score)
```

Markets are ranked by Final Score. The highest-ranked market is recommended.

---

## Data Source

**Live Market Data:** Real-time agricultural market prices from Data.gov.in Agmarknet API

**Crops:** Tomato, Onion, Potato, Rice, Wheat  
**Markets:** Bangalore, Mysore, Hubli, Belagavi, Mangalore  
**Storage:** MySQL table `market_live_data`

**Data Ingestion:**
- Automatic: Background scheduler fetches data hourly
- Manual: Use the "Live Market Data" page in the dashboard

**Retrain models:**

```bash
python -m ml.train_model
python -m ml.demand_training
python -m ml.risk_training
```

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `farmers` | Farmer accounts |
| `crops` | Farmer crop inventory |
| `markets` | Market details |
| `historical_prices` | ML training data |
| `predictions` | Stored ML predictions |
| `recommendations` | Market ranking results |

---

## Viva Demonstration Flow

1. **Home** — Show KPI cards and price trend charts
2. **Farmers** — Register/login a farmer
3. **Crops** — Add Tomato with quantity and production cost
4. **Markets** — Show 5 Karnataka markets
5. **Predictions** — Predict price for Tomato at Bangalore
6. **Recommendations** — Generate rankings, show best market
7. **Analytics** — Profit vs risk scatter, heatmap
8. **Reports** — Export CSV from Recommendations page

---

## Default Login (After Setup)

| Phone | Password |
|-------|----------|
| 9876543210 | farmer123 |
| 9876543211 | farmer123 |

---

## License

Academic project for final-year evaluation.
