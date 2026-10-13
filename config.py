"""
Application configuration for Market Selection and Profit Prediction System.
Data source: MySQL market_live_data (data.gov.in Agmarknet API).
No CSV datasets or synthetic data are used.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ML_DIR   = BASE_DIR / "ml"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
ML_DIR.mkdir(exist_ok=True)

# Model artifact paths (trained from market_live_data)
MODEL_PATH    = ML_DIR / "model.pkl"
ENCODERS_PATH = ML_DIR / "encoders.pkl"

# MySQL database configuration (override via .env)
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "farmer_market_db"

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ML configuration
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
TARGET_COLUMN = "future_price"

# Recommendation scoring weights
WEIGHT_PROFIT    = 0.60
WEIGHT_STABILITY = 0.25
WEIGHT_DISTANCE  = 0.15

# Fallback market list when database is unavailable
_DEFAULT_MARKETS = ["Bangalore", "Mysore", "Hubli", "Belagavi", "Mangalore"]


def get_markets() -> list:
    """Return distinct market names from market_live_data, or fallback list if DB unavailable."""
    try:
        from database.connection import test_connection, get_session
        from sqlalchemy import text

        test_connection()
        db = get_session()
        result = db.execute(
            text("SELECT DISTINCT market_name FROM market_live_data ORDER BY market_name")
        )
        markets = [row[0] for row in result if row[0]]
        db.close()
        if markets:
            return markets
    except Exception:
        pass
    return _DEFAULT_MARKETS


# Kept for backward compatibility
MARKETS = _DEFAULT_MARKETS

# Fallback crop list when database is unavailable
_DEFAULT_CROPS = ["Tomato", "Onion", "Potato", "Rice", "Wheat"]


def get_crops() -> list:
    """Return distinct crop names from market_live_data, or fallback list if DB unavailable."""
    try:
        from database.connection import test_connection, get_session
        from sqlalchemy import text

        test_connection()
        db = get_session()
        result = db.execute(
            text("SELECT DISTINCT crop_name FROM market_live_data ORDER BY crop_name")
        )
        crops = [row[0] for row in result if row[0]]
        db.close()
        if crops:
            return crops
    except Exception:
        pass
    return _DEFAULT_CROPS


# Kept for backward compatibility — all new code should use get_crops()
CROPS = _DEFAULT_CROPS

# Auto-refresh configuration
AUTO_REFRESH_ENABLED = os.getenv("AUTO_REFRESH_ENABLED", "true").lower() == "true"
_AUTO_REFRESH_INTERVAL = int(os.getenv("AUTO_REFRESH_INTERVAL_MINUTES", "3"))
AUTO_REFRESH_INTERVAL_MINUTES = max(3, min(15, _AUTO_REFRESH_INTERVAL))
