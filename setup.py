"""
Project setup script — Live Data Mode.
Bootstraps the system using real data from data.gov.in (Agmarknet API).
No synthetic data or CSV generation is used.

Run: python setup.py

Steps:
  1. bootstrap_live_data()  — fetch & store real records into market_live_data
  2. train_ml_model()       — train XGBoost price model on live data
  3. train_demand_model()   — train RandomForest demand model on live data
  4. train_risk_model()     — train RandomForest risk model on live data
  5. seed_database()        — seed farmers / markets / crops (DB only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Step 1: Bootstrap live data ───────────────────────────────────────────────

def bootstrap_live_data() -> None:
    """
    Fetch real market data from data.gov.in Agmarknet API and store in MySQL.
    Fetches all 5 configured crops for the last 30 days to build an initial
    training dataset with enough records for model training.

    Requires:
      - MySQL running and configured in .env
      - DATA_GOV_API_KEY set in .env
    """
    from database.connection import test_connection
    from services.live_market_service import (
        fetch_latest_market_data,
        store_market_data,
        get_live_record_count,
        _ensure_table,
    )
    import config

    print("\n[1/5] Bootstrapping live market data from data.gov.in...")

    if not test_connection():
        print("  ERROR: MySQL is unavailable.")
        print("  Please start MySQL, configure .env, and re-run setup.py")
        sys.exit(1)

    _ensure_table()
    existing = get_live_record_count()
    print(f"  Existing records in market_live_data: {existing}")

    total_inserted = 0
    # Fetch 30 days of history for all crops to build sufficient training data
    print("  Fetching last 30 days of market data...")
    records = fetch_latest_market_data(days_back=30)

    if not records:
        print("  ERROR: No records returned from API.")
        print("  Check your DATA_GOV_API_KEY in .env and internet connectivity.")
        sys.exit(1)

    inserted = store_market_data(records)
    total_inserted += inserted
    print(f"  Fetched {len(records)} records — inserted {inserted} new rows.")

    final_count = get_live_record_count()
    print(f"  Total records now in market_live_data: {final_count}")

    if final_count < 50:
        print(
            f"  WARNING: Only {final_count} records. "
            "The API may have limited records for Karnataka. "
            "Training will proceed but model quality may be limited."
        )


# ── Step 2–4: Train models ────────────────────────────────────────────────────

def train_ml_model() -> None:
    """Train XGBoost price prediction model using live DB data."""
    print("\n[2/5] Training XGBoost price prediction model...")
    from ml.train_model import main
    main()


def train_demand_model() -> None:
    """Train RandomForest demand model using live DB data."""
    print("\n[3/5] Training demand prediction model...")
    from ml.demand_training import main
    main()


def train_risk_model() -> None:
    """Train RandomForest risk classification model using live DB data."""
    print("\n[4/5] Training farmer risk model...")
    from ml.risk_training import main
    main()


# ── Step 5: Seed operational tables ──────────────────────────────────────────

def seed_database() -> None:
    """Seed farmers, markets, and crops into MySQL operational tables."""
    from database.connection import init_db, test_connection, SessionLocal
    from models.farmer import Farmer
    from models.market import Market
    from models.crop import Crop

    print("\n[5/5] Seeding operational database tables...")

    if not test_connection():
        print("  Skipping (DB unavailable).")
        return

    init_db()
    db = SessionLocal()
    try:
        if Farmer.count(db) == 0:
            f1 = Farmer.create(db, "Ramesh Kumar", "9876543210", "Mandya, Karnataka", "farmer123")
            f2 = Farmer.create(db, "Lakshmi Devi", "9876543211", "Hassan, Karnataka",  "farmer123")
            print(f"  Created farmers: {f1.name}, {f2.name}")
        else:
            print(f"  Farmers already exist ({Farmer.count(db)} records).")

        if Market.count(db) == 0:
            default_markets = [
                ("Bangalore",  "Bangalore APMC",         120, 2500, 38, 85, 70),
                ("Mysore",     "Mysore Mandi",            80, 1800, 32, 78, 65),
                ("Hubli",      "Hubli Wholesale",        200, 3200, 28, 72, 75),
                ("Belagavi",   "Belagavi Market",        280, 4500, 26, 68, 80),
                ("Mangalore",  "Mangalore Port Market",  350, 5200, 35, 82, 60),
            ]
            for m in default_markets:
                Market.create(db, *m)
            print("  Seeded 5 default markets.")
        else:
            print(f"  Markets already exist ({Market.count(db)} records).")

        if Crop.count(db) == 0:
            farmer = Farmer.get_by_phone(db, "9876543210")
            if farmer:
                Crop.create(db, farmer.farmer_id, "Tomato", 500, 15000, "Vegetable")
                Crop.create(db, farmer.farmer_id, "Onion",  300, 12000, "Vegetable")
                Crop.create(db, farmer.farmer_id, "Rice",   800,  8000, "Grain")
                Crop.create(db, farmer.farmer_id, "Wheat",  600,  7000, "Grain")
                Crop.create(db, farmer.farmer_id, "Potato", 400, 10000, "Vegetable")
                print("  Seeded 5 sample crops.")
        else:
            print(f"  Crops already exist ({Crop.count(db)} records).")

    finally:
        db.close()

    print("  Database seeding complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run complete live-data setup pipeline."""
    print("=" * 60)
    print("  AgriMarket AI — Live Data Setup")
    print("  Data source: data.gov.in Agmarknet API")
    print("=" * 60)

    bootstrap_live_data()
    train_ml_model()
    train_demand_model()
    train_risk_model()
    seed_database()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Run the app:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
