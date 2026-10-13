"""
Best Market Selection and Profit Prediction System for Farmers
Streamlit Dashboard Application — Premium UI Edition
"""
import asyncio
import atexit
import sys
import threading
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Global exception hooks (prevent ConnectionResetError from crashing the app) ──

def _handle_thread_exception(args):
    """Suppress benign ConnectionResetError and network errors in background threads."""
    if args.exc_type and issubclass(args.exc_type, (
        ConnectionResetError, ConnectionAbortedError, ConnectionRefusedError, OSError,
    )):
        return
    sys.stderr.write(f"Unhandled exception in thread {args.thread.name}: {args.exc_value}\n")

try:
    threading.excepthook = _handle_thread_exception
except AttributeError:
    pass


def _handle_asyncio_exception(loop, context):
    """Suppress ConnectionResetError in asyncio event loop callbacks."""
    exc = context.get("exception")
    if exc and isinstance(exc, (ConnectionResetError, ConnectionAbortedError, ConnectionRefusedError)):
        return
    loop.default_exception_handler(context)


def _install_asyncio_exception_handler():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
    try:
        loop.set_exception_handler(_handle_asyncio_exception)
    except (RuntimeError, AttributeError):
        pass


_NAV_KEYS = {
    "home": "nav_home",
    "farmer_dashboard": "nav_farmer_dashboard",
    "predictions": "nav_crop_price_prediction",
    "recommendations": "nav_recommendations",
    "analytics": "nav_analytics",
    "live_market": "nav_live_market_data",
    "demand": "nav_demand_prediction",
    "risk": "nav_risk_prediction",
    "trend": "nav_trend_forecast",
    "admin_dashboard": "nav_admin_dashboard",
    "farmers": "nav_farmer_management",
    "reports": "nav_reports",
    "history": "nav_prediction_history",
    "profile": "nav_profile",
}


def _nav_label(page_id):
    """Return translated navigation label for a canonical page id."""
    return t(_NAV_KEYS.get(page_id, page_id))


# ── End exception hooks ─────────────────────────────────────────────────────────


import config
from database.connection import SessionLocal, test_connection, init_db
from ml.predict import PricePredictor
from models.crop import Crop
from models.farmer import Farmer
from models.market import Market
from models.prediction import Prediction
from models.user import User
from models.prediction_history import PredictionHistory
from services.auth_service import (
    ensure_admin_exists,
    init_auth_tables,
    is_admin,
    is_farmer,
    is_authenticated,
    login,
    logout as auth_logout,
    register,
    require_role,
)
from services.recommendation_service import RecommendationService
from utils.helpers import format_currency
from utils.i18n import get_language, render_language_selector, set_language, t, t_level, t_msg
from utils.ui_components import (
    apply_plotly_theme,
    chart_card_close,
    chart_card_open,
    inject_theme,
    render_analytics_exec_summary,
    render_auto_refresh_status,
    render_best_market_hero,
    render_dashboard_header,
    render_kpi_cards,
    render_market_rank_cards,
    render_page_hero,
    render_report_cards,
    render_report_center_header,
    render_sidebar_branding,
    render_sidebar_footer,
    render_sidebar_status,
    style_bar_chart,
    style_line_chart,
)

# Dynamically resolve available crops and markets from the database
def _get_crops():
    return config.get_crops()

def _get_markets():
    return config.get_markets()

# Page config
st.set_page_config(
    page_title="AgriMarket AI | Farmer Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject premium theme (UI only)
inject_theme()


def get_db():
    """Get database session if available."""
    if not test_connection():
        return None
    return SessionLocal()


def init_session():
    """Initialize session state."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "farmer_id" not in st.session_state:
        st.session_state.farmer_id = None
    if "farmer_name" not in st.session_state:
        st.session_state.farmer_name = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "db_available" not in st.session_state:
        st.session_state.db_available = test_connection()


@st.cache_resource
def load_predictor():
    """Cache ML predictor."""
    return PricePredictor()


@st.cache_resource
def load_recommendation_service():
    """Cache recommendation service."""
    return RecommendationService()


def load_price_data() -> pd.DataFrame:
    """
    Load market data from MySQL market_live_data table.
    Maps live schema to analytics-compatible columns.
    Returns empty DataFrame if DB unavailable — UI shows guidance message.
    No CSV or synthetic fallback.
    """
    try:
        from services.live_market_service import get_training_data
        df = get_training_data()
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def page_home():
    """Home dashboard with KPI cards and charts."""
    render_dashboard_header(
        t("welcome_title"),
        t("welcome_subtitle"),
    )

    db = get_db()
    df = load_price_data()
    recs = st.session_state.recommendations
    best = recs[0] if recs else None

    farmer_count = Farmer.count(db) if db else 0
    market_count = Market.count(db) if db else 0
    crop_count = Crop.count(db) if db else 0
    best_market = best["market_name"] if best else "—"
    profit = format_currency(best["expected_profit"]) if best else "₹0.00"

    render_kpi_cards(
        [
            {"label": t("total_farmers"), "value": str(farmer_count), "icon": "👨‍🌾", "trend": "↑ " + t("registered_users")},
            {"label": t("total_markets"), "value": str(market_count), "icon": "🏪", "trend": "↑ " + t("active_mandis")},
            {"label": t("total_crops"), "value": str(crop_count), "icon": "🌱", "trend": "↑ " + t("crop_inventory")},
            {"label": t("best_market"), "value": best_market, "icon": "🏆", "trend": "★ " + t("top_recommendation")},
            {"label": t("predicted_profit"), "value": profit, "icon": "💰", "trend": "↑ " + t("expected_returns")},
        ]
    )

    # ── New ML prediction summary cards (shown only when results are available) ──
    new_cards = []
    # Last price prediction from Predictions page
    last_recs = st.session_state.get("recommendations", [])
    if last_recs:
        top_price = last_recs[0].get("predicted_price", 0)
        new_cards.append({
            "label": t("predicted_price"),
            "value": f"₹{top_price:,.2f}",
            "icon": "🔮",
            "trend": f"↑ {last_recs[0].get('market_name', '')}",
        })
    # Last demand prediction
    d_res = st.session_state.get("last_demand_result")
    if d_res:
        new_cards.append({
            "label": t("demand_prediction"),
            "value": f"{t_level(d_res['demand_level'])} ({d_res['demand_score']:.0f}%)",
            "icon": "📈",
            "trend": f"{st.session_state.get('last_demand_crop','')} · "
                     f"{st.session_state.get('last_demand_market','')}",
        })
    # Last risk prediction
    r_res = st.session_state.get("last_risk_result")
    if r_res:
        new_cards.append({
            "label": t("risk_prediction"),
            "value": f"{t_level(r_res['risk_level'])} ({r_res['risk_score']:.0f}%)",
            "icon": "⚠️",
            "trend": t("ml_risk_assessment"),
        })
    if new_cards:
        render_kpi_cards(new_cards)

    # ── Live Data Status Section ──
    st.markdown("---")
    chart_card_open(t("live_data_status"), "📡")
    try:
        from services.live_market_service import get_live_record_count, get_latest_market_data
        from services.scheduler_service import get_status as sched_status
        
        total_records = get_live_record_count()
        sched_status = sched_status()
        
        # Get date range and commodity/market counts
        live_df = get_latest_market_data(days=365, limit=50000)
        if not live_df.empty:
            earliest_date = live_df["market_date"].min()
            latest_date = live_df["market_date"].max()
            total_commodities = live_df["crop_name"].nunique()
            total_markets = live_df["market_name"].nunique()
        else:
            earliest_date = "N/A"
            latest_date = "N/A"
            total_commodities = 0
            total_markets = 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t("total_live_records"), f"{total_records:,}")
            st.metric(t("earliest_record"), str(earliest_date))
        with col2:
            st.metric(t("latest_record"), str(latest_date))
            st.metric(t("total_commodities"), total_commodities)
        with col3:
            st.metric(t("last_api_sync"), sched_status["last_fetch"])
            st.metric(t("total_markets"), total_markets)
    except Exception as e:
        st.warning(t("unable_live_status", error=e))
    chart_card_close()

    c1, c2 = st.columns(2)

    with c1:
        chart_card_open(t("crop_price_trends"), "📈")
        if not df.empty:
            trend = df.groupby(["crop_name", "month"])["future_price"].mean().reset_index()
            fig = px.line(
                trend,
                x="month",
                y="future_price",
                color="crop_name",
                title=t("avg_future_price_month"),
                labels={"future_price": t("price_rupee"), "month": t("month")},
            )
            fig.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(style_line_chart(fig), width="stretch")
        else:
            st.info(t("no_live_data_yet"))
        chart_card_close()

    with c2:
        chart_card_open(t("market_ranking"), "🏆")
        if recs:
            rank_df = pd.DataFrame(recs)
            fig = px.bar(
                rank_df,
                x="market_name",
                y="final_score",
                color="risk_level",
                title=t("market_recommendation_scores"),
                labels={"final_score": t("final_score")},
                color_discrete_sequence=["#0F9D58", "#34A853", "#8BC34A", "#F9A825", "#E53935"],
            )
            st.plotly_chart(style_bar_chart(fig), width="stretch")
        else:
            st.info(t("generate_recs_hint"))
        chart_card_close()

    if recs:
        chart_card_open(t("profit_comparison"), "💰")
        profit_df = pd.DataFrame(recs)
        fig2 = px.bar(
            profit_df,
            x="market_name",
            y="expected_profit",
            text="expected_profit",
            title=t("expected_profit_market"),
            color="market_name",
            color_discrete_sequence=["#0F9D58", "#34A853", "#8BC34A", "#66BB6A", "#AED581"],
        )
        fig2.update_traces(texttemplate="₹%{text:.0f}", textposition="outside")
        st.plotly_chart(style_bar_chart(fig2), width="stretch")
        chart_card_close()


def page_farmers():
    """Farmer registration, login, and profile."""
    render_page_hero("👨‍🌾", t("farmer_management"), t("farmer_mgmt_subtitle"))
    tab1, tab2, tab3 = st.tabs(["📝 " + t("register"), "🔐 " + t("login"), "👤 " + t("profile")])

    db = get_db()

    with tab1:
        st.markdown('<div class="form-glass">', unsafe_allow_html=True)
        with st.form("register_form"):
            name = st.text_input(t("full_name"))
            phone = st.text_input(t("phone_number"))
            location = st.text_input(t("location"))
            password = st.text_input(t("password"), type="password")
            submitted = st.form_submit_button(t("register"), type="primary")
            if submitted:
                if db and name and phone and location and password:
                    existing = Farmer.get_by_phone(db, phone)
                    if existing:
                        st.error(t("phone_exists"))
                    else:
                        farmer = Farmer.create(db, name, phone, location, password)
                        st.success(t("registered_success", farmer_id=farmer.farmer_id))
                else:
                    st.warning(t("fill_all_fields"))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="form-glass">', unsafe_allow_html=True)
        with st.form("login_form"):
            phone = st.text_input(t("phone_number"), key="login_phone")
            password = st.text_input(t("password"), type="password", key="login_pass")
            login_btn = st.form_submit_button(t("login"), type="primary")
            if login_btn:
                if db:
                    farmer = Farmer.get_by_phone(db, phone)
                    if farmer and farmer.verify_password(password):
                        st.session_state.logged_in = True
                        st.session_state.farmer_id = farmer.farmer_id
                        st.session_state.farmer_name = farmer.name
                        st.success(t("welcome_user", name=farmer.name))
                    else:
                        st.error(t("invalid_credentials"))
                else:
                    st.error(t("db_required_use"))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        if st.session_state.logged_in:
            st.success(t("logged_in_as", name=st.session_state.farmer_name))
            if db and st.session_state.farmer_id:
                farmer = Farmer.get_by_id(db, st.session_state.farmer_id)
                if farmer:
                    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
                    with st.form("profile_form"):
                        new_name = st.text_input(t("name"), value=farmer.name)
                        new_loc = st.text_input(t("location"), value=farmer.location)
                        new_pass = st.text_input(t("new_password_optional"), type="password")
                        if st.form_submit_button(t("update_profile"), type="primary"):
                            Farmer.update_profile(
                                db,
                                farmer.farmer_id,
                                name=new_name,
                                location=new_loc,
                                password=new_pass or None,
                            )
                            st.session_state.farmer_name = new_name
                            st.success(t("profile_updated"))
                    st.markdown("</div>", unsafe_allow_html=True)
            if st.button(t("logout_farmer")):
                auth_logout()
                st.rerun()
        else:
            st.warning(t("please_login_first"))

    if db:
        st.markdown("#### 📋 " + t("all_farmers"))
        farmers = Farmer.get_all(db)
        if farmers:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            t("id"): f.farmer_id,
                            t("name"): f.name,
                            t("phone"): f.phone,
                            t("location"): f.location,
                        }
                        for f in farmers
                    ]
                ),
                width="stretch",
            )
        db.close()


def page_crops():
    """Crop management."""
    render_page_hero("🌱", t("crop_management"), t("crop_mgmt_subtitle"))
    db = get_db()

    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
    with st.form("add_crop"):
        col1, col2 = st.columns(2)
        with col1:
            crop_name = st.selectbox(t("crop_name"), _get_crops())
            quantity = st.number_input(t("quantity"), min_value=1.0, value=100.0)
        with col2:
            production_cost = st.number_input(t("production_cost"), min_value=0.0, value=5000.0)
            category = st.selectbox(t("category"), [t("vegetable"), t("grain"), t("pulses"), t("other")])
        farmer_id = st.session_state.farmer_id or 1
        if st.form_submit_button(t("add_crop"), type="primary"):
            if db:
                Crop.create(db, farmer_id, crop_name, quantity, production_cost, category)
                st.success(t("added_crop", crop=crop_name))
            else:
                st.error(t("db_required_crops"))
    st.markdown("</div>", unsafe_allow_html=True)

    if db:
        crops = Crop.get_all(db)
        if crops:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            t("id"): c.crop_id,
                            t("farmer"): c.farmer_id,
                            t("crop"): c.crop_name,
                            t("quantity"): float(c.quantity),
                            t("cost"): float(c.production_cost),
                            t("category"): c.category,
                        }
                        for c in crops
                    ]
                ),
                width="stretch",
            )
        db.close()


def page_markets():
    """Market management."""
    render_page_hero("🏪", t("market_management"), t("market_mgmt_subtitle"))
    db = get_db()

    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
    with st.form("add_market"):
        c1, c2 = st.columns(2)
        with c1:
            market_name = st.text_input(t("market_name"))
            market_location = st.text_input(t("market_location"))
            distance = st.number_input(t("distance"), min_value=1.0, value=100.0)
        with c2:
            transport = st.number_input(t("transportation_cost"), min_value=0.0, value=2000.0)
            price = st.number_input(t("current_market_price"), min_value=0.0, value=30.0)
            demand = st.slider(t("demand_level"), 0, 100, 70)
            supply = st.slider(t("supply_level"), 0, 100, 60)
        if st.form_submit_button(t("add_market"), type="primary"):
            if db and market_name:
                Market.create(
                    db,
                    market_name,
                    market_location,
                    distance,
                    transport,
                    price,
                    demand,
                    supply,
                )
                st.success(t("market_added", name=market_name))
            else:
                st.warning(t("market_required"))
    st.markdown("</div>", unsafe_allow_html=True)

    if db:
        markets = Market.get_all(db)
        if markets:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            t("id"): m.market_id,
                            t("name"): m.market_name,
                            t("location_col"): m.market_location,
                            t("distance"): float(m.distance_km),
                            t("transport"): float(m.transportation_cost),
                            t("price"): float(m.current_market_price),
                            t("demand"): float(m.demand_level),
                            t("supply"): float(m.supply_level),
                        }
                        for m in markets
                    ]
                ),
                width="stretch",
            )
        else:
            st.info(t("no_markets_db"))
        db.close()
    else:
        st.info(t("db_required_markets"))


def _auto_fetch_features(crop_name: str, market_name: str):
    """
    Automatically fetch historical_price, demand_level, supply_level
    from market_live_data for the given crop and market.

    Uses the same demand estimation logic as the training pipeline
    (_map_to_training_schema) to keep feature distributions consistent.

    Returns dict with keys: historical_price, demand_level, supply_level
    Returns None if no historical data is available.
    """
    try:
        from services.live_market_service import get_latest_market_data

        df = get_latest_market_data(
            crop_name=crop_name,
            market_name=market_name,
            days=365,
            limit=1,
            latest_only=True,
        )

        if df.empty:
            return None

        record = df.iloc[0]
        historical_price = float(record.get("modal_price", 0) or 0)
        if historical_price <= 0:
            return None

        arrival_qty = float(record.get("arrival_quantity", 0) or 0)

        # Demand: min-max scaled modal_price per crop (35-90 range, same as training)
        crop_df = get_latest_market_data(
            crop_name=crop_name,
            days=365,
            limit=50000,
        )
        if not crop_df.empty and "modal_price" in crop_df.columns:
            prices = pd.to_numeric(crop_df["modal_price"], errors="coerce").dropna()
            prices = prices[prices > 0]
            if len(prices) > 1:
                mn, mx = float(prices.min()), float(prices.max())
                if mx > mn:
                    demand_level = ((historical_price - mn) / (mx - mn) * 55 + 35)
                else:
                    demand_level = 60.0
            else:
                demand_level = 60.0
        else:
            demand_level = 60.0
        demand_level = round(max(10.0, min(95.0, demand_level)), 2)

        # Supply: arrival_quantity scaled to 0-100, or fallback formula
        if arrival_qty > 0:
            supply_level = min(90.0, max(15.0, (arrival_qty / 10000.0) * 100.0))
        else:
            supply_level = 100.0 - demand_level * 0.65
        supply_level = round(max(15.0, min(90.0, supply_level)), 2)

        return {
            "historical_price": historical_price,
            "demand_level": demand_level,
            "supply_level": supply_level,
        }
    except Exception:
        return None


def page_predictions():
    """ML price predictions."""
    render_page_hero("🔮", t("price_predictions"), t("price_pred_subtitle"))
    predictor = load_predictor()
    metrics = predictor.get_metrics()

    render_kpi_cards(
        [
            {"label": t("mae"), "value": str(metrics.get("mae", "N/A")), "icon": "📉"},
            {"label": t("rmse"), "value": str(metrics.get("rmse", "N/A")), "icon": "📊"},
            {"label": t("r2_score"), "value": str(metrics.get("r2", "N/A")), "icon": "🎯"},
        ]
    )

    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
    st.markdown("**" + t("predict_future_price") + "**")
    c1, c2, c3 = st.columns(3)
    with c1:
        crop = st.selectbox(t("crop"), _get_crops(), key="pred_crop")
        market = st.selectbox(t("market"), _get_markets(), key="pred_market")
    with c2:
        month = st.number_input(t("month"), 1, 12, datetime.now().month)
        distance = st.number_input(t("distance_km"), 10.0, 500.0, 150.0)
    with c3:
        transport = st.number_input(t("transport_cost"), 0.0, 10000.0, 2500.0)

    if st.button(t("predict_price"), type="primary"):
        features = _auto_fetch_features(crop, market)
        if features is None:
            st.error(t("no_historical_data"))
        else:
            record = {
                "crop_name": crop,
                "market_name": market,
                "month": int(month),
                "historical_price": features["historical_price"],
                "demand_level": features["demand_level"],
                "supply_level": features["supply_level"],
                "transportation_cost": transport,
                "distance_km": distance,
            }
            predicted = predictor.predict_single(record)
            st.success(
                f"**{t('predicted_future_price')}**\n\n"
                f"{t('crop_colon')} **{crop}**\n\n"
                f"{t('market_colon')} **{market}**\n\n"
                f"**{t('predicted_modal_price')}**\n\n"
                f"**{format_currency(predicted)} {t('per_quintal')}**"
            )

            db = get_db()
            if db:
                mkt = Market.get_by_name(db, market)
                if mkt:
                    Prediction.create(db, mkt.market_id, crop, market, predicted)
                db.close()
            _save_prediction_history("price", crop, market, record, {"predicted_price": predicted})
    st.markdown("</div>", unsafe_allow_html=True)


def page_recommendations():
    """Market recommendation engine."""
    render_page_hero("✅", "Market Recommendations", "AI-ranked markets by profit, stability, and distance")
    service = load_recommendation_service()

    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        crop = st.selectbox(t("select_crop"), _get_crops())
        quantity = st.number_input(t("quantity"), min_value=1.0, value=500.0)
    with c2:
        production_cost = st.number_input(t("production_cost"), min_value=0.0, value=15000.0)
        month = st.number_input(t("target_month"), 1, 12, datetime.now().month)
    st.markdown("</div>", unsafe_allow_html=True)

    db = get_db()
    markets = []
    if db:
        db_markets = Market.get_all(db)
        markets = [
            {
                "market_id": m.market_id,
                "market_name": m.market_name,
                "market_location": m.market_location,
                "distance_km": float(m.distance_km),
                "transportation_cost": float(m.transportation_cost),
                "current_market_price": float(m.current_market_price),
                "demand_level": float(m.demand_level),
                "supply_level": float(m.supply_level),
            }
            for m in db_markets
        ]
        db.close()

    if not markets:
        # DB markets table not seeded yet — derive defaults from live market data
        from services.live_market_service import get_latest_market_data
        live_df = get_latest_market_data(days=90, limit=2000)
        from services.live_market_service import _logistics
        for m_name in _get_markets():
            dist_km, transport = _logistics(m_name)
            if not live_df.empty:
                subset = live_df[live_df["market_name"].str.contains(
                    m_name, case=False, na=False
                )]
                price = float(subset["modal_price"].mean()) if len(subset) else 3000.0
            else:
                price = 3000.0
            markets.append({
                "market_name":          m_name,
                "distance_km":          float(dist_km),
                "transportation_cost":  float(transport),
                "current_market_price": price,
                "demand_level":         70.0,
                "supply_level":         60.0,
            })
        if not markets:
            st.warning(t("no_market_data"))
            return

    if st.button(t("generate_recommendations"), type="primary"):
        with st.spinner(t("analyzing_markets")):
            recs = service.generate_recommendations(
                crop_name=crop,
                quantity=quantity,
                production_cost=production_cost,
                markets=markets,
                month=int(month),
                farmer_id=st.session_state.farmer_id,
                crop_id=1,
            )
            st.session_state.recommendations = recs

    recs = st.session_state.recommendations
    if recs:
        best = service.get_best_market(recs)
        render_best_market_hero(best)
        render_market_rank_cards(recs)

        display_df = pd.DataFrame(
            [
                {
                    t("rank"): r["market_rank"],
                    t("market"): r["market_name"],
                    t("predicted_price"): r["predicted_price"],
                    t("predicted_profit"): r["expected_profit"],
                    t("risk_level_col"): t_level(r["risk_level"]),
                    t("final_score"): r["final_score"],
                }
                for r in recs
            ]
        )
        st.dataframe(display_df, width="stretch")

        chart_card_open(t("profit_vs_score"), "📊")
        fig = px.bar(
            display_df,
            x=t("market"),
            y=t("predicted_profit"),
            color=t("final_score"),
            title=t("market_profit_comparison"),
            color_continuous_scale=["#8BC34A", "#34A853", "#0F9D58"],
        )
        st.plotly_chart(style_bar_chart(fig), width="stretch")
        chart_card_close()

        render_report_center_header()
        render_report_cards()
        c1, c2 = st.columns(2)
        with c1:
            if st.button(t("export_profit_report"), type="primary", width="stretch"):
                path = service.export_profit_report(recs)
                st.success(t("saved", path=path))
        with c2:
            if st.button(t("export_ranking_report"), type="primary", width="stretch"):
                path = service.export_ranking_report(recs)
                st.success(t("saved", path=path))


def page_analytics():
    """Analytics and risk visualization."""
    render_page_hero("📊", "Analytics & Risk Analysis", "Executive-level insights and performance intelligence")
    df = load_price_data()
    recs = st.session_state.recommendations
    render_analytics_exec_summary(len(df), len(recs))

    tab1, tab2, tab3 = st.tabs([t("price_analytics"), t("risk_analysis"), t("model_info")])

    with tab1:
        if not df.empty:
            crop_filter = st.multiselect(
                t("filter_crops"), _get_crops(), default=_get_crops()[:3]
            )
            filtered = df[df["crop_name"].isin(crop_filter)]
            chart_card_open(t("price_distribution"), "📦")
            fig = px.box(
                filtered,
                x="crop_name",
                y="future_price",
                color="market_name",
                title=t("price_distribution_title"),
                color_discrete_sequence=["#0F9D58", "#34A853", "#8BC34A", "#66BB6A", "#AED581"],
            )
            st.plotly_chart(apply_plotly_theme(fig), width="stretch")
            chart_card_close()

            heatmap = filtered.pivot_table(
                values="future_price",
                index="crop_name",
                columns="market_name",
                aggfunc="mean",
            )
            chart_card_open(t("avg_price_heatmap"), "🗺️")
            fig2 = px.imshow(
                heatmap,
                title=t("avg_price_heatmap"),
                labels=dict(color=t("price_rupee")),
                color_continuous_scale=["#F5F8F6", "#8BC34A", "#34A853", "#0F9D58", "#0B1F14"],
            )
            st.plotly_chart(apply_plotly_theme(fig2, height=420), width="stretch")
            chart_card_close()
        else:
            st.info(t("no_live_data_analytics"))

    with tab2:
        if recs:
            risk_df = pd.DataFrame(recs)
            c1, c2 = st.columns(2)
            with c1:
                chart_card_open(t("profit_vs_risk"), "⚡")
                fig = px.scatter(
                    risk_df,
                    x="expected_profit",
                    y="risk_score",
                    size="final_score",
                    color="market_name",
                    title=t("profit_vs_risk_title"),
                    labels={"expected_profit": t("predicted_profit"), "risk_score": t("risk_score")},
                    color_discrete_sequence=["#0F9D58", "#34A853", "#8BC34A", "#F9A825", "#E53935"],
                )
                st.plotly_chart(apply_plotly_theme(fig), width="stretch")
                chart_card_close()
            with c2:
                chart_card_open(t("risk_distribution"), "🎯")
                risk_df["risk_level"] = risk_df["risk_level"].map(t_level)
                fig2 = px.pie(
                    risk_df,
                    names="risk_level",
                    title=t("risk_level_distribution"),
                    color_discrete_sequence=["#0F9D58", "#F9A825", "#E53935"],
                )
                fig2.update_traces(textposition="inside", pull=[0.05, 0, 0])
                st.plotly_chart(apply_plotly_theme(fig2), width="stretch")
                chart_card_close()
        else:
            st.info(t("generate_recs_risk"))

    with tab3:
        st.markdown('<div class="form-glass">', unsafe_allow_html=True)
        predictor = load_predictor()
        metrics = predictor.get_metrics()
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric(t("mae"), metrics.get("mae", "N/A"))
        with mc2:
            st.metric(t("rmse"), metrics.get("rmse", "N/A"))
        with mc3:
            st.metric("R²", metrics.get("r2", "N/A"))
        st.markdown(
            f"""
            **{t("algorithm")}** XGBoost Regressor  
            **{t("features")}** Crop, Market, Month, Historical Price, Demand, Supply,  
            Transportation Cost, Distance + engineered features  
            **{t("scoring")}** 60% Profit + 25% Stability + 15% Distance
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


def page_live_market():
    """NEW — Real-Time Agricultural Market Data (Agmarknet Integration)."""
    render_page_hero(
        "📡",
        t("live_market_data"),
        t("live_market_subtitle"),
    )

    # ── Status cards ─────────────────────────────────────────────────────────
    from services.live_market_service import (
        get_live_record_count,
        fetch_latest_market_data,
        store_market_data,
        get_latest_market_data,
    )
    from services.scheduler_service import get_status as sched_status

    count   = get_live_record_count()
    sched   = sched_status()

    render_kpi_cards(
        [
            {
                "label": t("live_records_db"),
                "value": f"{count:,}",
                "icon": "🗄️",
                "trend": t("live_table"),
            },
            {
                "label": t("last_fetch"),
                "value": sched["last_fetch"],
                "icon": "🕐",
                "trend": t("auto_refresh_every", minutes=sched.get('interval_minutes', 3)),
            },
            {
                "label": t("last_retrain"),
                "value": sched["last_retrain"],
                "icon": "🤖",
                "trend": t("after_new_records", count=100),
            },
            {
                "label": t("scheduler"),
                "value": "🟢 " + t("running") if sched["running"] else "⚪ " + t("idle"),
                "icon": "⚙️",
                "trend": t("next_sync", time=sched.get('next_sync', '—')),
            },
        ]
    )

    # ── Live Sync Status ─────────────────────────────────────────────────────
    st.markdown("---")
    chart_card_open(t("real_time_sync_status"), "🔄")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric(t("last_sync_time"), sched["last_fetch"])
    with sc2:
        st.metric(t("next_scheduled_sync"), sched.get("next_sync", "—"))
    with sc3:
        status_icon = "🟢" if sched["running"] else "⚪"
        st.metric(t("scheduler_status"), f"{status_icon} {t('running') if sched['running'] else t('idle')}")
    with sc4:
        st.metric(t("records_synced_today"), sched.get("records_today", 0))
    if st.button(t("run_sync_now"), type="primary", key="btn_sync_now", width="stretch"):
        from services.scheduler_service import run_once
        inserted = run_once()
        if inserted > 0:
            st.success(t("sync_complete_new", count=inserted))
        else:
            st.info(t("sync_complete_none"))
        st.rerun()
    chart_card_close()

    # ── API Key status ────────────────────────────────────────────────────────
    from data_sources.agmarknet_fetcher import get_api_key
    if get_api_key():
        st.success(t("api_key_configured"))
    else:
        st.error(t("api_key_not_set"))

    # ── Manual fetch button ───────────────────────────────────────────────────
    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
    st.markdown("**" + t("fetch_store_data") + "**")
    c1, c2 = st.columns(2)
    with c1:
        fetch_days = st.slider(t("days_of_history"), 1, 30, 7, key="live_days")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t("fetch_now"), type="primary", key="btn_fetch_live"):
            with st.spinner(t("fetching_data")):
                try:
                    records = fetch_latest_market_data(days_back=fetch_days)
                    inserted = store_market_data(records)
                    if inserted > 0:
                        st.success(t("fetched_inserted", fetched=len(records), inserted=inserted))
                    elif records:
                        st.info(t("fetched_duplicates", count=len(records)))
                    else:
                        st.warning(t("no_records_returned"))
                except Exception as exc:
                    st.error(t("fetch_failed", error=exc))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Live data table ───────────────────────────────────────────────────────
    st.markdown("---")
    chart_card_open(t("latest_market_prices"), "📊")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        date_range = st.selectbox(
            t("date_range"),
            [t("latest"), t("today"), t("last_7_days"), t("last_30_days"), t("custom_date")],
            key="live_date_range",
        )
    with filter_col2:
        view_crop = st.selectbox(
            t("filter_by_crop"), [t("all")] + _get_crops(), key="live_view_crop"
        )
    with filter_col3:
        view_market = st.selectbox(
            t("filter_by_market"), [t("all")] + _get_markets(), key="live_view_market"
        )

    custom_date_val = None
    if date_range == t("custom_date"):
        custom_date_val = st.date_input(t("select_date"), value=date.today(), key="live_custom_date")

    # Map date range selection to query params
    latest_only = False
    start_date = None
    end_date = None
    if date_range == t("latest"):
        latest_only = True
    elif date_range == t("today"):
        td = date.today()
        start_date = td.isoformat()
        end_date = td.isoformat()
    elif date_range == t("last_7_days"):
        start_date = (date.today() - timedelta(days=7)).isoformat()
        end_date = date.today().isoformat()
    elif date_range == t("last_30_days"):
        start_date = (date.today() - timedelta(days=30)).isoformat()
        end_date = date.today().isoformat()
    elif date_range == t("custom_date"):
        cd = custom_date_val or date.today()
        start_date = cd.isoformat()
        end_date = cd.isoformat()

    try:
        df = get_latest_market_data(
            crop_name=None if view_crop == t("all") else view_crop,
            market_name=None if view_market == t("all") else view_market,
            days=30,
            limit=500,
            latest_only=latest_only,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty and date_range == t("today"):
            st.info(t("no_today_records"))
            try:
                new_records = fetch_latest_market_data(days_back=1)
                if new_records:
                    stored = store_market_data(new_records)
                    st.info(t("fetched_stored", fetched=len(new_records), stored=stored))
                else:
                    st.info(t("api_no_today"))
            except Exception:
                st.info(t("api_fetch_failed_latest"))
            df = get_latest_market_data(
                crop_name=None if view_crop == t("all") else view_crop,
                market_name=None if view_market == t("all") else view_market,
                days=30,
                limit=500,
                latest_only=True,
            )
            if not df.empty:
                st.info(t("today_not_available"))

        if not df.empty:
            show_cols = [
                c for c in [
                    "crop_name", "market_name", "state", "district",
                    "modal_price", "min_price", "max_price",
                    "arrival_quantity", "market_date",
                ]
                if c in df.columns
            ]
            st.dataframe(df[show_cols], width="stretch")

            if "modal_price" in df.columns and "market_date" in df.columns:
                try:
                    chart_df = df.copy()
                    chart_df["market_date"] = pd.to_datetime(
                        chart_df["market_date"], errors="coerce"
                    )
                    chart_df = chart_df.dropna(subset=["market_date", "modal_price"])
                    if not chart_df.empty:
                        chart_card_open(t("modal_price_trend"), "📈")
                        fig = px.line(
                            chart_df.sort_values("market_date"),
                            x="market_date",
                            y="modal_price",
                            color="crop_name",
                            title=t("live_modal_price_trend"),
                            labels={"modal_price": t("modal_price"), "market_date": t("date")},
                        )
                        st.plotly_chart(
                            style_line_chart(fig), width="stretch"
                        )
                        chart_card_close()
                except Exception:
                    pass
        else:
            st.info(t("no_live_db"))
    except Exception as exc:
        st.error(t("could_not_load_live", error=exc))

    chart_card_close()

    # ── Scheduler controls ────────────────────────────────────────────────────
    st.markdown("---")
    chart_card_open(t("background_scheduler"), "⏰")
    from services import scheduler_service

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button(t("start_auto_refresh"), key="btn_sched_start"):
            scheduler_service.start()
            interval = scheduler_service.get_status().get("interval_minutes", 3)
            st.success(t("scheduler_started", minutes=interval))
    with sc2:
        if st.button(t("stop_scheduler"), key="btn_sched_stop"):
            scheduler_service.stop()
            st.info(t("scheduler_stopped"))

    s = scheduler_service.get_status()
    cols = st.columns(4)
    with cols[0]:
        st.metric(t("auto_refresh_metric"), "🟢 " + t("running") if s.get("running") else "🔴 " + t("stopped"))
    with cols[1]:
        st.metric(t("refresh_interval"), f"{s.get('interval_minutes', 3)} min")
    with cols[2]:
        st.metric(t("last_refresh"), s.get("last_fetch", "—"))
    with cols[3]:
        st.metric(t("next_refresh"), s.get("next_sync", "—"))
    cols2 = st.columns(4)
    with cols2[0]:
        st.metric(t("last_status"), s.get("last_refresh_status", "—"))
    with cols2[1]:
        st.metric(t("records_fetched"), s.get("records_today", 0))
    with cols2[2]:
        from services.live_market_service import get_live_record_count
        st.metric(t("total_records"), get_live_record_count())
    with cols2[3]:
        st.metric(t("duration"), s.get("last_duration", "—"))
    chart_card_close()

    # ── Setup guidance ────────────────────────────────────────────────────────
    with st.expander(t("how_to_configure")):
        _interval = config.AUTO_REFRESH_INTERVAL_MINUTES
        st.markdown(
            f"""
            {t("setup_instructions")}

            {t("step1")}
            {t("step1_1")}
            {t("step1_2")}
            {t("step1_3")}

            {t("step2")}
            ```
            DATA_GOV_API_KEY=your_api_key_here
            ```

            {t("step3")}
            ```bash
            python setup.py
            ```
            {t("step3_1")}
            {t("step3_2")}

            {t("step4")}
            {t("step4_1", minutes=_interval)}
            {t("step4_2")}

            {t("api_unavailable")}
            {t("api_unavailable_1")}
            {t("api_unavailable_2")}

            {t("data_source")}
            {t("data_source_1")}
            {t("data_source_2")}
            """
        )

    # ── Auto-refresh timer ──────────────────────────────────────────────────
    # Re-renders the page every 30 seconds so status values update automatically
    # when the background scheduler fires (every 3 minutes).
    import time as _time
    _now = _time.time()
    _last = st.session_state.get("page_live_refresh_ts", None)
    if _last is None:
        st.session_state.page_live_refresh_ts = _now
    elif _now - _last > 30:
        st.session_state.page_live_refresh_ts = _now
        st.rerun()


def page_demand_prediction():
    """NEW — Market Demand Prediction using RandomForest ML model."""
    render_page_hero(
        "📈",
        t("market_demand_prediction"),
        t("demand_pred_subtitle"),
    )

    # Lazy-load the service (cached at session level to avoid reload on rerun)
    if "demand_service" not in st.session_state:
        from services.demand_service import DemandService
        st.session_state.demand_service = DemandService()

    demand_svc = st.session_state.demand_service

    # ── Model metrics strip ──────────────────────────────────────────────────
    try:
        d_metrics = demand_svc.get_model_metrics()
        render_kpi_cards(
            [
                {"label": t("mae"),      "value": str(d_metrics.get("mae",  "N/A")), "icon": "📉"},
                {"label": t("rmse"),     "value": str(d_metrics.get("rmse", "N/A")), "icon": "📊"},
                {"label": t("r2_score"), "value": str(d_metrics.get("r2",   "N/A")), "icon": "🎯"},
            ]
        )
    except Exception:
        pass

    # ── Input form ───────────────────────────────────────────────────────────
    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
    st.markdown("**" + t("predict_future_demand") + "**")
    c1, c2 = st.columns(2)
    with c1:
        d_crop   = st.selectbox(t("crop"),   _get_crops(),   key="dem_crop")
        d_market = st.selectbox(t("market"), _get_markets(), key="dem_market")
    with c2:
        d_month  = st.number_input(t("month"), 1, 12, datetime.now().month, key="dem_month")

    if st.button(t("predict_demand"), type="primary", key="btn_predict_demand"):
        with st.spinner(t("predicting_demand")):
            try:
                features = _auto_fetch_features(d_crop, d_market)
                if features is None:
                    st.error(t("no_historical_data"))
                else:
                    result = demand_svc.predict_market_demand(
                        crop_name=d_crop,
                        market_name=d_market,
                        month=int(d_month),
                        historical_price=features["historical_price"],
                        demand_level=features["demand_level"],
                        supply_level=features["supply_level"],
                    )
                    score = result["demand_score"]
                    level = result["demand_level"]

                    # Save to session for dashboard display
                    st.session_state["last_demand_result"] = result
                    st.session_state["last_demand_crop"]   = d_crop
                    st.session_state["last_demand_market"] = d_market

                    # Result display
                    level_colors = {"Low": "#F9A825", "Medium": "#34A853", "High": "#0F9D58"}
                    color = level_colors.get(level, "#34A853")
                    st.markdown(
                        f"""
                        <div style="background:rgba(255,255,255,0.5);border-radius:12px;
                             padding:1.5rem;margin-top:1rem;text-align:center;
                             border:2px solid {color};">
                            <div style="font-size:2.2rem;font-weight:700;color:{color};">
                                {score:.1f} / 100
                            </div>
                            <div style="font-size:1.1rem;color:#0B1F14;margin-top:0.3rem;">
                                {t('demand_level_colon', level=f'<strong style="color:{color};">{t_level(level)}</strong>')}
                            </div>
                            <div style="font-size:0.85rem;color:#5a6b62;margin-top:0.5rem;">
                                {d_crop} · {d_market} · {t("month")} {int(d_month)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Persist to DB if available
                    db = get_db()
                    if db:
                        try:
                            from sqlalchemy import text
                            db.execute(
                                text(
                                    "INSERT INTO market_demand_predictions "
                                    "(crop_name, market_name, month, demand_score, demand_level, "
                                    "historical_price, demand_input, supply_input) "
                                    "VALUES (:cn,:mn,:mo,:ds,:dl,:hp,:di,:si)"
                                ),
                                dict(
                                    cn=d_crop, mn=d_market, mo=int(d_month),
                                    ds=score, dl=level,
                                    hp=features["historical_price"],
                                    di=features["demand_level"],
                                    si=features["supply_level"],
                                ),
                            )
                            db.commit()
                        except Exception:
                            pass
                        finally:
                            db.close()

                    _save_prediction_history("demand", d_crop, d_market,
                        {"month": int(d_month)},
                        {"demand_score": score, "demand_level": level})

            except Exception as exc:
                st.error(t("prediction_failed", error=exc))

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Bulk comparison: all markets ─────────────────────────────────────────
    st.markdown("---")
    chart_card_open(t("demand_comparison"), "🏪")
    if st.button(t("compare_all_markets"), key="btn_demand_all"):
        features = _auto_fetch_features(d_crop, d_market)
        if features is None:
            st.error(t("no_historical_data"))
        else:
            df = load_price_data()
            markets_list = []
            for mkt in _get_markets():
                subset = df[df["market_name"] == mkt] if not df.empty else pd.DataFrame()
                markets_list.append(
                    {
                        "market_name": mkt,
                        "demand_level": float(subset["demand_level"].mean()) if len(subset) else 70.0,
                        "supply_level": float(subset["supply_level"].mean()) if len(subset) else 60.0,
                    }
                )
            try:
                all_results = demand_svc.predict_all_markets(
                    crop_name=d_crop,
                    month=int(d_month),
                    historical_price=features["historical_price"],
                    markets=markets_list,
                )
                comp_df = pd.DataFrame(all_results)
                fig = px.bar(
                    comp_df,
                    x="market_name",
                    y="demand_score",
                    color="demand_level",
                    text="demand_score",
                    title=t("predicted_demand_score", crop=d_crop),
                    labels={"demand_score": t("demand_score_label"), "market_name": t("market")},
                    color_discrete_map={"Low": "#F9A825", "Medium": "#34A853", "High": "#0F9D58"},
                )
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                st.plotly_chart(style_bar_chart(fig), width="stretch")
            except Exception as exc:
                st.error(t("comparison_failed", error=exc))
    chart_card_close()


def page_risk_prediction():
    """NEW — Farmer Risk Prediction using RandomForestClassifier."""
    render_page_hero(
        "⚠️",
        t("farmer_risk_prediction"),
        t("risk_pred_subtitle"),
    )

    # Lazy-load both new services
    if "risk_ml_service" not in st.session_state:
        from services.risk_ml_service import RiskMLService
        st.session_state.risk_ml_service = RiskMLService()
    if "demand_service" not in st.session_state:
        from services.demand_service import DemandService
        st.session_state.demand_service = DemandService()

    risk_svc   = st.session_state.risk_ml_service
    demand_svc = st.session_state.demand_service

    # ── Model accuracy strip ─────────────────────────────────────────────────
    try:
        r_metrics = risk_svc.get_model_metrics()
        render_kpi_cards(
            [
                {"label": t("accuracy"),       "value": str(r_metrics.get("accuracy", "N/A")), "icon": "🎯"},
                {"label": t("train_samples"),  "value": str(r_metrics.get("train_samples", "N/A")), "icon": "📦"},
                {"label": t("classes"),        "value": "3", "icon": "🏷️"},
            ]
        )
    except Exception:
        pass

    # ── Input form ───────────────────────────────────────────────────────────
    st.markdown('<div class="form-glass">', unsafe_allow_html=True)
    st.markdown("**" + t("predict_risk_title") + "**")
    c1, c2 = st.columns(2)
    with c1:
        r_crop   = st.selectbox(t("crop"),   _get_crops(),   key="risk_crop")
        r_market = st.selectbox(t("market"), _get_markets(), key="risk_market")
    with c2:
        r_month  = st.number_input(t("month"), 1, 12, datetime.now().month, key="risk_month")

    if st.button(t("predict_risk"), type="primary", key="btn_predict_risk"):
        with st.spinner(t("analysing_risk")):
            try:
                features = _auto_fetch_features(r_crop, r_market)
                if features is None:
                    st.error(t("no_historical_data"))
                else:
                    # Get logistics for this market
                    from services.live_market_service import _logistics
                    dist_km, transport_cost = _logistics(r_market)

                    predictor_svc = load_predictor()
                    price_record = {
                        "crop_name":          r_crop,
                        "market_name":        r_market,
                        "month":              int(r_month),
                        "historical_price":   features["historical_price"],
                        "demand_level":       features["demand_level"],
                        "supply_level":       features["supply_level"],
                        "transportation_cost": transport_cost,
                        "distance_km":        dist_km,
                    }
                    predicted_price = predictor_svc.predict_single(price_record)

                    # Demand score from demand model
                    d_result = demand_svc.predict_market_demand(
                        crop_name=r_crop,
                        market_name=r_market,
                        month=int(r_month),
                        historical_price=features["historical_price"],
                        demand_level=features["demand_level"],
                        supply_level=features["supply_level"],
                    )
                    demand_score = d_result["demand_score"]

                    # Risk prediction
                    risk_result = risk_svc.predict_from_context(
                        predicted_price=predicted_price,
                        historical_price=features["historical_price"],
                        demand_level=features["demand_level"],
                        supply_level=features["supply_level"],
                        demand_score=demand_score,
                    )
                    r_score = risk_result["risk_score"]
                    r_level = risk_result["risk_level"]

                    # Save to session
                    st.session_state["last_risk_result"]        = risk_result
                    st.session_state["last_risk_predicted_price"] = predicted_price
                    st.session_state["last_demand_for_risk"]    = d_result

                    # Risk color map
                    r_colors = {
                        "Low Risk":    "#0F9D58",
                        "Medium Risk": "#F9A825",
                        "High Risk":   "#E53935",
                    }
                    r_color = r_colors.get(r_level, "#34A853")

                    # Display summary
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(
                            f"""
                            <div style="background:rgba(255,255,255,0.5);border-radius:12px;
                                 padding:1.2rem;text-align:center;border:2px solid #0F9D58;">
                                <div style="font-size:0.8rem;color:#5a6b62;">{t("predicted_price")}</div>
                                <div style="font-size:1.8rem;font-weight:700;color:#0F9D58;">
                                    ₹{predicted_price:,.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_b:
                        d_color = {"Low": "#F9A825", "Medium": "#34A853", "High": "#0F9D58"}.get(
                            d_result["demand_level"], "#34A853"
                        )
                        st.markdown(
                            f"""
                            <div style="background:rgba(255,255,255,0.5);border-radius:12px;
                                 padding:1.2rem;text-align:center;border:2px solid {d_color};">
                                <div style="font-size:0.8rem;color:#5a6b62;">{t("demand_prediction")}</div>
                                <div style="font-size:1.8rem;font-weight:700;color:{d_color};">
                                    {t_level(d_result['demand_level'])}
                                </div>
                                <div style="font-size:0.8rem;color:#5a6b62;">
                                    ({demand_score:.1f}%)
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_c:
                        st.markdown(
                            f"""
                            <div style="background:rgba(255,255,255,0.5);border-radius:12px;
                                 padding:1.2rem;text-align:center;border:2px solid {r_color};">
                                <div style="font-size:0.8rem;color:#5a6b62;">{t("risk_prediction")}</div>
                                <div style="font-size:1.8rem;font-weight:700;color:{r_color};">
                                    {t_level(r_level)}
                                </div>
                                <div style="font-size:0.8rem;color:#5a6b62;">
                                    ({r_score:.1f}%)
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Persist to DB if available
                    db = get_db()
                    if db:
                        try:
                            from sqlalchemy import text as sa_text
                            vol = abs(predicted_price - features["historical_price"]) / max(features["historical_price"], 1.0)
                            fluc = abs(features["demand_level"] - features["supply_level"]) / 100.0
                            pm = (predicted_price - features["historical_price"]) / max(features["historical_price"], 1.0)
                            db.execute(
                                sa_text(
                                    "INSERT INTO farmer_risk_predictions "
                                    "(predicted_price, historical_price_volatility, market_fluctuation, "
                                    "demand_score, supply_level, profit_margin, risk_score, risk_level) "
                                    "VALUES (:pp,:hv,:mf,:ds,:sl,:pm,:rs,:rl)"
                                ),
                                dict(
                                    pp=predicted_price, hv=round(vol, 4),
                                    mf=round(fluc, 4), ds=demand_score,
                                    sl=features["supply_level"], pm=round(pm, 4),
                                    rs=r_score, rl=r_level,
                                ),
                            )
                            db.commit()
                        except Exception:
                            pass
                        finally:
                            db.close()

                    _save_prediction_history("risk", r_crop, r_market,
                        {"month": int(r_month)},
                        {"predicted_price": predicted_price, "risk_score": r_score,
                         "risk_level": r_level, "demand_score": demand_score})

            except Exception as exc:
                st.error(t("risk_pred_failed", error=exc))

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Risk guidance panel ──────────────────────────────────────────────────
    st.markdown("---")
    chart_card_open(t("risk_guide"), "📋")
    st.markdown(
        "\n".join([
            t("risk_guide_header"),
            t("risk_guide_row1", low_risk=t_level("Low Risk")),
            t("risk_guide_row2", medium_risk=t_level("Medium Risk")),
            t("risk_guide_row3", high_risk=t_level("High Risk")),
        ]),
        unsafe_allow_html=False,
    )
    chart_card_close()


def page_trend_forecast():
    """Market Trend Forecasting page using time-series regression."""
    render_page_hero("📉", t("market_trend_forecast"),
                     t("forecast_subtitle"))

    from services.forecast_service import FORECAST_PERIODS, run_forecast, save_forecast

    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox(t("select_crop"), _get_crops(), key="fc_crop")
    with col2:
        market = st.selectbox(t("select_market"), _get_markets(), key="fc_market")
    with col3:
        period = st.selectbox(t("forecast_period"), list(FORECAST_PERIODS.keys()), key="fc_period")

    if st.button(t("run_forecast"), type="primary", width="stretch"):
        with st.spinner(t("training_forecast", crop=crop, market=market)):
            result = run_forecast(crop, market, period)
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(
                    t("forecast_complete",
                      trend=result['trend_direction'],
                      confidence=result['confidence'],
                      r2=result['r2'],
                      rmse=result['rmse'])
                )
                save_forecast(crop, market, result)

                # Chart
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=result["historical_dates"],
                        y=result["historical_prices"],
                        mode="lines+markers",
                        name=t("historical"),
                        line=dict(color="#0F9D58", width=2),
                        marker=dict(size=5, color="#0F9D58"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=result["forecast_dates"],
                        y=result["forecast_prices"],
                        mode="lines+markers+text",
                        name=t("forecast_label", period=period),
                        line=dict(color="#FF6D00", width=2, dash="dash"),
                        marker=dict(size=6, color="#FF6D00"),
                        text=[f"₹{p:.0f}" for p in result["forecast_prices"]],
                        textposition="top center",
                        textfont=dict(size=10, color="#FF6D00"),
                    )
                )
                fig.update_layout(height=450)
                st.plotly_chart(apply_plotly_theme(fig, height=450), width="stretch")

                # Summary metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                direction_icon = {"Upward": "📈", "Downward": "📉", "Stable": "➡️"}
                with col_a:
                    st.metric(t("trend_direction"), f"{direction_icon.get(result['trend_direction'], '➡️')} {t_level(result['trend_direction'])}")
                with col_b:
                    st.metric(t("confidence"), f"{result['confidence']}%")
                with col_c:
                    avg_fc = sum(result["forecast_prices"]) / len(result["forecast_prices"])
                    st.metric(t("avg_forecast_price"), f"₹{avg_fc:.0f}")
                with col_d:
                    last_hist = result["historical_prices"][-1] if result["historical_prices"] else 0
                    change_pct = ((avg_fc - last_hist) / max(last_hist, 1)) * 100
                    st.metric(t("expected_change"), f"{change_pct:+.1f}%")

                # Explanation
                st.info(t("forecast_explanation"))

                _save_prediction_history("forecast", crop, market,
                    {"period": period},
                    {"trend_direction": result.get("trend_direction"),
                     "confidence": result.get("confidence"),
                     "r2": result.get("r2"), "rmse": result.get("rmse"),
                     "avg_forecast_price": avg_fc})


def page_community():
    """Farmer Community discussion portal."""
    from services.community_service import (
        add_comment,
        create_post,
        get_all_posts,
        get_comments,
        get_post_by_id,
    )

    render_page_hero("💬", t("farmer_community"), t("community_subtitle"))

    tab_view, tab_new = st.tabs([t("browse_discussions"), t("new_post")])

    with tab_new:
        st.markdown("### " + t("create_new_discussion"))
        title = st.text_input(t("title"), placeholder=t("title_placeholder"))
        message = st.text_area(t("message"), height=150, placeholder=t("message_placeholder"))
        author = st.text_input(t("your_name"), placeholder=t("anonymous"), value=st.session_state.get("farmer_name", ""))
        if st.button(t("submit_post"), type="primary", width="stretch"):
            if not title.strip() or not message.strip():
                st.error(t("title_message_required"))
            else:
                post_id = create_post(title.strip(), message.strip(), author.strip() or t("anonymous"))
                if post_id:
                    st.success(t("post_published"))
                    st.rerun()
                else:
                    st.error(t("post_create_failed"))

    with tab_view:
        search = st.text_input(t("search_discussions"), placeholder=t("search_placeholder"))
        posts = get_all_posts(search.strip() if search else "")

        if not posts:
            st.info(t("no_discussions"))
        else:
            for post in posts:
                with st.expander(
                    f"**{post['title']}**  —  🧑‍🌾 {post['author_name']}  "
                    f"| 🕐 {post['created_at'].strftime('%d %b %Y, %I:%M %p')} "
                    f"| 💬 {post['comment_count']} {t('comments')}",
                    expanded=False,
                ):
                    st.markdown(post["message"])
                    st.markdown("---")
                    st.markdown(t("comments_colon"))
                    comments = get_comments(post["id"])
                    if comments:
                        for c in comments:
                            st.markdown(
                                f"<div style='background:#f0fdf4;padding:8px 12px;"
                                f"border-radius:8px;margin:4px 0;'>"
                                f"<strong>{c['author_name']}</strong> "
                                f"<span style='color:#888;font-size:0.8em;'>"
                                f"({c['created_at'].strftime('%d %b %Y, %I:%M %p')})</span><br>"
                                f"{c['comment']}</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption(t("no_comments"))
                    with st.form(key=f"comment_form_{post['id']}"):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            c_author = col_b.text_input(
                                t("your_name"),
                                value=st.session_state.get("farmer_name", ""),
                                key=f"ca_{post['id']}",
                                label_visibility="collapsed",
                                placeholder=t("your_name_ph"),
                            )
                            comment_text = st.text_area(
                                t("comment"),
                                key=f"ct_{post['id']}",
                                label_visibility="collapsed",
                                placeholder=t("write_comment"),
                            )
                        submitted = st.form_submit_button(t("reply"), width="stretch")
                        if submitted and comment_text.strip():
                            ok = add_comment(
                                post["id"],
                                comment_text.strip(),
                                c_author.strip() or t("anonymous"),
                            )
                            if ok:
                                st.success(t("comment_added"))
                                st.rerun()
                            else:
                                st.error(t("comment_failed"))


def _save_prediction_history(pred_type: str, crop_name: str, market_name: str, input_data: dict, result: dict):
    """Save prediction record to history if user is authenticated. Extracts flat values from result dict."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    db = get_db()
    if db:
        try:
            predicted_price = result.get("predicted_price") if isinstance(result, dict) else None
            demand_score = result.get("demand_score") if isinstance(result, dict) else None
            risk_level = result.get("risk_level") if isinstance(result, dict) else None
            PredictionHistory.create(
                db, user_id=user_id, prediction_type=pred_type,
                crop_name=crop_name, market_name=market_name,
                predicted_price=predicted_price,
                demand_score=demand_score,
                risk_level=risk_level,
                input_data=input_data, result=result,
            )
        except Exception:
            pass
        finally:
            db.close()


# ── Login Page ────────────────────────────────────────────────────────────────

def page_login():
    """Login and registration page with role selection."""
    init_auth_tables()
    ensure_admin_exists()

    # Apply login-page class to body for background
    st.markdown("""<div class="login-page-root">""", unsafe_allow_html=True)

    # Inject login-specific styles inline (only present on login page)
    st.markdown("""<style>
/* ── Full-page centering via flexbox on the main block container ── */
header[data-testid="stHeader"]{display:none!important}
[data-testid="stSidebar"]{display:none!important}
.block-container{display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;min-height:100vh!important;padding:0!important;max-width:none!important}
.stVerticalBlock{gap:0!important}
.block-container > div{max-width:440px!important;width:100%!important;margin-left:auto!important;margin-right:auto!important;padding:0!important}

/* ── Glass card for tabs (seamless continuation of header) ── */
.stTabs{position:relative;z-index:9999;width:100%;background:rgba(255,255,255,0.12);backdrop-filter:blur(24px) saturate(1.4);-webkit-backdrop-filter:blur(24px) saturate(1.4);border:1px solid rgba(255,255,255,0.25);border-top:none;border-radius:0 0 24px 24px;padding:0.75rem 2.5rem 2.5rem;box-shadow:0 24px 80px rgba(0,0,0,0.35),0 0 0 1px rgba(255,255,255,0.1)}

/* ── Tab buttons (high contrast) ── */
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.10)!important;border-radius:14px!important;padding:0.25rem!important}
.stTabs [data-baseweb="tab"]{color:#ffffff!important;font-size:0.82rem!important;border-radius:10px!important;padding:0.4rem 0.8rem!important;font-weight:500!important}
.stTabs [aria-selected="true"]{background:rgba(255,255,255,0.2)!important;color:#ffffff!important;font-weight:600!important}

/* ── Form labels (pure white for max contrast) ── */
.stTabs label{color:#ffffff!important;font-weight:600!important;margin-bottom:0.35rem!important}
.stTabs .stMarkdown h5{color:#ffffff!important;font-weight:600!important;margin-bottom:1.35rem!important}

/* ── Input fields ── */
.stTabs .stTextInput{margin-bottom:1rem!important}
.stTabs .stTextInput > div > div > input{background:rgba(255,255,255,0.12)!important;border:1.5px solid rgba(255,255,255,0.2)!important;color:#ffffff!important;border-radius:10px!important;padding:0.7rem 0.9rem!important;font-size:0.88rem!important}
.stTabs .stTextInput > div > div > input::placeholder{color:rgba(255,255,255,0.65)!important}
.stTabs .stTextInput > div > div > input:focus{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(255,193,7,0.15)!important;background:rgba(255,255,255,0.18)!important}

/* ── Submit button (margin only, no visual changes) ── */
.stTabs .stButton > button[kind="primary"]{margin-top:0.5rem!important}

/* ── Alerts ── */
.stTabs .stAlert{background:rgba(255,255,255,0.10)!important;border:1px solid rgba(255,255,255,0.15)!important;border-radius:10px!important;color:#ffffff!important}
</style>""", unsafe_allow_html=True)

    # Logo + title + card header (self-contained — no unclosed divs wrapping Streamlit widgets)
    st.markdown(
        f"""
        <div class="login-card-header">
            <div class="login-logo-icon">🌾</div>
            <div class="login-title">AgriMarket AI</div>
            <div class="login-subtitle">{t("smart_agriculture_analytics")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_language_selector()

    tab_admin, tab_farmer, tab_register = st.tabs([t("admin_login"), t("farmer_login"), t("register_tab")])

    # ── Admin Login Tab ──────────────────────────────────────────────
    with tab_admin:
        with st.form("auth_admin_form"):
            st.markdown("##### " + t("sign_in_admin"))
            l_user = st.text_input(t("username"), key="auth_admin_user")
            l_pass = st.text_input(t("password"), type="password", key="auth_admin_pass")
            if st.form_submit_button(t("sign_in_as_admin"), type="primary", width="stretch"):
                success, msg, user = login(l_user, l_pass)
                if success:
                    if user.role != "ADMIN":
                        st.error(t("not_admin"))
                    else:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user.user_id
                        st.session_state.username = user.username
                        st.session_state.user_role = user.role
                        if user.farmer_id:
                            db = get_db()
                            if db:
                                try:
                                    farmer = Farmer.get_by_id(db, user.farmer_id)
                                    if farmer:
                                        st.session_state.farmer_id = farmer.farmer_id
                                        st.session_state.farmer_name = farmer.name
                                except Exception:
                                    pass
                                finally:
                                    db.close()
                        st.session_state["nav_to"] = "admin_dashboard"
                        st.rerun()
                else:
                    st.error(t_msg(msg))

    # ── Farmer Login Tab ─────────────────────────────────────────────
    with tab_farmer:
        with st.form("auth_farmer_form"):
            st.markdown("##### " + t("sign_in_farmer"))
            l_user = st.text_input(t("username"), key="auth_farmer_user")
            l_pass = st.text_input(t("password"), type="password", key="auth_farmer_pass")
            if st.form_submit_button(t("sign_in_as_farmer"), type="primary", width="stretch"):
                success, msg, user = login(l_user, l_pass)
                if success:
                    if user.role != "FARMER":
                        st.error(t("not_farmer"))
                    else:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user.user_id
                        st.session_state.username = user.username
                        st.session_state.user_role = user.role
                        if user.farmer_id:
                            db = get_db()
                            if db:
                                try:
                                    farmer = Farmer.get_by_id(db, user.farmer_id)
                                    if farmer:
                                        st.session_state.farmer_id = farmer.farmer_id
                                        st.session_state.farmer_name = farmer.name
                                except Exception:
                                    pass
                                finally:
                                    db.close()
                        st.session_state["nav_to"] = "farmer_dashboard"
                        st.rerun()
                else:
                    st.error(t_msg(msg))

    # ── Register Tab ─────────────────────────────────────────────────
    with tab_register:
        with st.form("auth_register_form"):
            st.markdown("##### " + t("create_farmer_account"))
            r_user = st.text_input(t("choose_username"))
            r_email = st.text_input(t("email_optional"))
            r_pass = st.text_input(t("choose_password"), type="password")
            r_pass2 = st.text_input(t("confirm_password"), type="password")
            if st.form_submit_button(t("register"), type="primary", width="stretch"):
                if r_pass != r_pass2:
                    st.error(t("passwords_no_match"))
                else:
                    success, msg = register(r_user, r_pass, role="FARMER", email=r_email or None)
                    if success:
                        st.success(t_msg(msg))
                        st.info(t("can_now_sign_in"))
                    else:
                        st.error(t_msg(msg))

    st.markdown(
        f'<div class="login-footer-text">{t("login_footer")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Admin Pages ───────────────────────────────────────────────────────────────

def require_auth():
    """Redirect to login if user is not authenticated."""
    if not st.session_state.get("logged_in", False):
        st.warning(t("please_login_first"))
        st.stop()

def require_admin():
    """Redirect if user is not admin."""
    require_auth()
    if st.session_state.get("user_role") != "ADMIN":
        st.error(t("access_denied"))
        st.stop()

def page_admin_dashboard():
    """Admin dashboard with system overview, stats, and prediction analytics."""
    require_admin()
    render_page_hero("🛡️", t("nav_admin_dashboard"), t("admin_dash_subtitle"))
    db = get_db()

    if db:
        try:
            from pathlib import Path
            user_count = User.count(db)
            farmer_count = Farmer.count(db)
            from services.scheduler_service import get_status as api_status
            from services.live_market_service import get_live_record_count
            live_records = get_live_record_count()
            history_count = PredictionHistory.count_all(db)
            from models.market import Market
            market_count = Market.count(db) if hasattr(Market, 'count') else 0
            admin_count = len(User.get_all_by_role(db, "ADMIN"))

            # KPI cards row
            render_kpi_cards([
                {"label": t("total_users"), "value": str(user_count), "icon": "👥", "trend": t("admins_count", count=admin_count)},
                {"label": t("total_farmers"), "value": str(farmer_count), "icon": "👨‍🌾", "trend": t("farmer_profiles_short")},
                {"label": t("total_predictions"), "value": str(history_count), "icon": "🔮", "trend": t("all_time")},
                {"label": t("live_records"), "value": f"{live_records:,}", "icon": "📊", "trend": t("agmarknet_data")},
                {"label": t("total_markets"), "value": str(market_count), "icon": "🏪", "trend": t("active_mandis")},
            ])

            # ── System Status ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### " + t("system_status"))
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                api_ok = live_records > 0
                st.metric(t("agmarknet_api"), t("connected") if api_ok else t("no_data"),
                          help=t("live_records_synced", count=live_records))
            with scol2:
                db_ok = test_connection()
                st.metric(t("database"), t("connected") if db_ok else t("disconnected"))
            with scol3:
                ml_dir = Path(__file__).resolve().parent / "ml"
                price_ok = (ml_dir / "model.pkl").exists()
                demand_ok = (ml_dir / "demand_model.pkl").exists()
                risk_ok = (ml_dir / "risk_model.pkl").exists()
                ml_status = f"Price ✓ Demand ✓ Risk ✓" if (price_ok and demand_ok and risk_ok) else t("partial")
                ml_icon = "✅" if (price_ok and demand_ok and risk_ok) else "⚠️"
                st.metric(t("ml_models"), f"{ml_icon} {ml_status}")

            # ── Prediction Statistics ─────────────────────────────────────
            st.markdown("---")
            st.markdown("#### " + t("prediction_statistics"))
            pcols = st.columns(4)
            ptype_labels = {"price": t("price_type"), "demand": t("demand_type"), "risk": t("risk_type"), "forecast": t("forecast_type")}
            for i, ptype in enumerate(["price", "demand", "risk", "forecast"]):
                with pcols[i]:
                    cnt = PredictionHistory.count_by_type(db, ptype)
                    st.metric(ptype_labels[ptype], cnt)

            # ── Most Predicted Crops ──────────────────────────────────────
            st.markdown("#### " + t("most_predicted_crops"))
            top_crops = PredictionHistory.get_most_predicted_crops(db, limit=5)
            if top_crops:
                crop_cols = st.columns(len(top_crops))
                for i, entry in enumerate(top_crops):
                    with crop_cols[i]:
                        st.metric(entry["crop"], entry["count"])
            else:
                st.info(t("no_predictions"))

            # ── Recent Activity ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### " + t("recent_activity"))
            act_col1, act_col2 = st.columns(2)

            with act_col1:
                st.markdown(t("recent_users"))
                all_users = User.get_all(db)
                recent_users = all_users[-5:][::-1] if all_users else []
                if recent_users:
                    for u in recent_users:
                        created = u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "—"
                        st.markdown(f"- {u.username} ({u.role}) — {created}")
                else:
                    st.info(t("no_users"))

            with act_col2:
                st.markdown(t("recent_predictions"))
                recent_preds = PredictionHistory.get_all(db, limit=5)
                if recent_preds:
                    for r in recent_preds:
                        ts = r.prediction_timestamp.strftime("%Y-%m-%d %H:%M") if r.prediction_timestamp else "—"
                        st.markdown(f"- {r.prediction_type} | {r.crop_name or '—'} | {ts}")
                else:
                    st.info(t("no_predictions"))

            # ── Recent Predictions Table ───────────────────────────────────
            st.markdown("---")
            st.markdown("#### " + t("recent_predictions_table"))
            recent = PredictionHistory.get_all(db, limit=20)
            if recent:
                rows = []
                for r in recent:
                    rows.append({
                        t("id"): r.prediction_id,
                        t("user"): r.user_id,
                        t("type"): r.prediction_type,
                        t("crop"): r.crop_name or "—",
                        t("market"): r.market_name or "—",
                        t("price"): f"₹{r.predicted_price:.2f}" if r.predicted_price else "—",
                        t("demand"): f"{r.demand_score:.1f}" if r.demand_score else "—",
                        t("risk_prediction"): r.risk_level or "—",
                        t("time"): r.prediction_timestamp.strftime("%Y-%m-%d %H:%M") if r.prediction_timestamp else "—",
                    })
                st.dataframe(rows, width="stretch")
            else:
                st.info(t("no_prediction_history"))
        except Exception as e:
            st.error(t("admin_dash_error", error=e))
        finally:
            db.close()


def page_admin_farmers():
    """Admin panel for managing farmer accounts."""
    require_admin()
    render_page_hero("👨‍🌾", t("nav_farmer_management"), t("farmer_mgmt_subtitle_admin"))
    db = get_db()

    if db:
        try:
            # Show all users
            st.markdown("#### " + t("all_users"))
            users = User.get_all(db)
            if users:
                rows = []
                for u in users:
                    rows.append({
                        t("id"): u.user_id,
                        t("username"): u.username,
                        t("role"): u.role,
                        t("farmer_id"): u.farmer_id or "—",
                        t("created"): u.created_at.strftime("%Y-%m-%d") if u.created_at else "—",
                    })
                st.dataframe(rows, width="stretch")
            else:
                st.info(t("no_users_found"))

            st.markdown("---")
            st.markdown("#### " + t("farmer_profiles"))
            farmers = Farmer.get_all(db)
            if farmers:
                frows = []
                for f in farmers:
                    frows.append({
                        t("id"): f.farmer_id,
                        t("name"): f.name,
                        t("phone"): f.phone,
                        t("location_col"): f.location,
                        t("created"): f.created_at.strftime("%Y-%m-%d") if f.created_at else "—",
                    })
                st.dataframe(frows, width="stretch")
            else:
                st.info(t("no_farmer_profiles"))

            # Create farmer user
            st.markdown("---")
            st.markdown("#### " + t("create_farmer_user"))
            with st.form("admin_create_farmer"):
                c1, c2 = st.columns(2)
                with c1:
                    new_username = st.text_input(t("username"))
                    new_password = st.text_input(t("password"), type="password")
                with c2:
                    new_name = st.text_input(t("farmer_name"))
                    new_phone = st.text_input(t("phone"))
                    new_location = st.text_input(t("location"))
                if st.form_submit_button(t("create_farmer"), type="primary"):
                    if new_username and new_password and new_name:
                        # Check existing
                        existing_user = User.get_by_username(db, new_username)
                        if existing_user:
                            st.error(t("username_exists"))
                        else:
                            # Create farmer profile first
                            farmer = Farmer.create(db, new_name, new_phone or "0000000000", new_location or "", new_password)
                            # Create user linked to farmer
                            User.create(db, username=new_username, password=new_password, role="FARMER", farmer_id=farmer.farmer_id)
                            st.success(t("farmer_created", name=new_name, username=new_username))
                    else:
                        st.warning(t("fields_required"))
        except Exception as e:
            st.error(t("error_colon", error=e))
        finally:
            db.close()


def page_admin_reports():
    """Admin reports and data export."""
    require_admin()
    render_page_hero("📁", t("reports_page"), t("reports_subtitle"))
    db = get_db()

    if db:
        try:
            st.markdown("#### " + t("system_summary"))
            from services.live_market_service import get_live_record_count
            live_count = get_live_record_count()
            user_count = User.count(db)
            farmer_count = Farmer.count(db)
            history_count = PredictionHistory.count_all(db)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(t("total_live_records"), f"{live_count:,}")
                st.metric(t("total_users"), user_count)
            with col2:
                st.metric(t("total_farmers"), farmer_count)
                st.metric(t("total_predictions"), history_count)

            st.markdown("---")
            st.markdown("#### " + t("export_options"))
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button(t("export_users_csv"), width="stretch"):
                    users = User.get_all(db)
                    import pandas as pd
                    df = pd.DataFrame([{
                        "user_id": u.user_id, "username": u.username,
                        "role": u.role, "farmer_id": u.farmer_id,
                        "created_at": str(u.created_at),
                    } for u in users])
                    csv = df.to_csv(index=False)
                    st.download_button(t("download_users_csv"), csv, "users.csv", "text/csv")
            with ec2:
                if st.button(t("export_pred_history_csv"), width="stretch"):
                    records = PredictionHistory.get_all(db, limit=1000)
                    import pandas as pd
                    df = pd.DataFrame([{
                        "prediction_id": r.prediction_id, "user_id": r.user_id,
                        "type": r.prediction_type, "crop": r.crop_name,
                        "market": r.market_name, "price": float(r.predicted_price) if r.predicted_price else "",
                        "demand_score": float(r.demand_score) if r.demand_score else "",
                        "risk_level": r.risk_level or "",
                        "timestamp": str(r.prediction_timestamp),
                    } for r in records])
                    csv = df.to_csv(index=False)
                    st.download_button(t("download_predictions_csv"), csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(t("error_colon", error=e))
        finally:
            db.close()


# ── Farmer Dashboard ──────────────────────────────────────────────────────────

def page_farmer_dashboard():
    """Farmer dashboard showing personal predictions, quick links, and profile."""
    require_auth()
    render_page_hero("🌾", t("farmer_dashboard"),
                     t("farmer_dash_subtitle"))
    db = get_db()

    username = st.session_state.get("username", "Farmer")
    farmer_name = st.session_state.get("farmer_name", username)
    user_id = st.session_state.get("user_id")

    # Welcome + profile summary
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(t("welcome_name", name=farmer_name))
        st.markdown(t("username_role", username=username))
    with c2:
        if st.button(t("edit_profile"), width="stretch"):
            st.session_state["nav_to"] = "profile"
            st.rerun()

    # ── My Prediction Stats ─────────────────────────────────────────────
    if db and user_id:
        try:
            summary = PredictionHistory.get_user_summary(db, user_id)
            total_preds = sum(summary.values())
            render_kpi_cards([
                {"label": t("my_predictions"), "value": str(total_preds), "icon": "🔮", "trend": t("total")},
                {"label": t("price_predictions"), "value": str(summary.get("price", 0)), "icon": "💰", "trend": ""},
                {"label": t("demand_prediction"), "value": str(summary.get("demand", 0)), "icon": "📈", "trend": ""},
                {"label": t("risk_prediction"), "value": str(summary.get("risk", 0)), "icon": "⚠️", "trend": ""},
            ])
        except Exception:
            pass

        # ── Recent Predictions ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### " + t("my_recent_predictions"))
        my_preds = PredictionHistory.get_by_user(db, user_id, limit=10)
        if my_preds:
            rows = []
            for r in my_preds:
                rows.append({
                    t("id"): r.prediction_id, t("type"): r.prediction_type,
                    t("crop"): r.crop_name or "—", t("market"): r.market_name or "—",
                    t("price"): f"₹{r.predicted_price:.2f}" if r.predicted_price else "—",
                    t("demand"): f"{r.demand_score:.1f}" if r.demand_score else "—",
                    t("risk_prediction"): r.risk_level or "—",
                    t("time"): r.prediction_timestamp.strftime("%Y-%m-%d %H:%M") if r.prediction_timestamp else "—",
                })
            st.dataframe(rows, width="stretch")
        else:
            st.info(t("no_predictions_yet"))
        db.close()
    else:
        st.info(t("db_unavailable_history"))

    # ── Quick Links ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### " + t("quick_access"))
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.markdown("##### 💰 " + t("price_predictions"))
        st.markdown(t("predict_future_xgb"))
        if st.button(t("go_price_prediction"), key="q_price"):
            st.session_state["nav_to"] = "predictions"
            st.rerun()
    with qc2:
        st.markdown("##### 📈 " + t("demand_prediction"))
        st.markdown(t("forecast_rf"))
        if st.button(t("go_demand_prediction"), key="q_demand"):
            st.session_state["nav_to"] = "demand"
            st.rerun()
    with qc3:
        st.markdown("##### ⚠️ " + t("risk_prediction"))
        st.markdown(t("assess_risk"))
        if st.button(t("go_risk_prediction"), key="q_risk"):
            st.session_state["nav_to"] = "risk"
            st.rerun()


def page_prediction_history():
    """Show prediction history for all users (admin) or current user (farmer)."""
    require_auth()
    render_page_hero("📋", t("nav_prediction_history"), t("pred_hist_subtitle"))
    db = get_db()
    if not db:
        st.info(t("db_unavailable_short"))
        return
    user_id = st.session_state.get("user_id")
    is_admin_user = st.session_state.get("user_role") == "ADMIN"
    try:
        if is_admin_user:
            st.markdown("#### " + t("all_predictions_admin"))
            records = PredictionHistory.get_all(db, limit=200)
        else:
            st.markdown("#### " + t("my_predictions_heading"))
            records = PredictionHistory.get_by_user(db, user_id, limit=200)
        if records:
            rows = []
            for r in records:
                rows.append({
                    t("id"): r.prediction_id, t("type"): r.prediction_type,
                    t("crop"): r.crop_name or "—", t("market"): r.market_name or "—",
                    t("price"): f"₹{r.predicted_price:.2f}" if r.predicted_price else "—",
                    t("demand"): f"{r.demand_score:.1f}" if r.demand_score else "—",
                    t("risk_prediction"): r.risk_level or "—",
                    t("time"): r.prediction_timestamp.strftime("%Y-%m-%d %H:%M") if r.prediction_timestamp else "—",
                })
            st.dataframe(rows, width="stretch")
        else:
            st.info(t("no_predictions_found"))
    except Exception as exc:
        st.error(t("could_not_load_history", error=exc))
    finally:
        db.close()


def page_profile():
    """User profile page with account info and edit capability."""
    require_auth()
    render_page_hero("👤", t("my_profile"), t("profile_subtitle"))
    db = get_db()
    if not db:
        st.info(t("db_unavailable_short"))
        return
    user_id = st.session_state.get("user_id")
    username = st.session_state.get("username", "User")
    role = st.session_state.get("user_role", "")
    try:
        user = User.get_by_id(db, user_id) if user_id else None
        farmer = None
        if user and user.farmer_id:
            farmer = Farmer.get_by_id(db, user.farmer_id)

        st.markdown("#### " + t("profile_information"))

        cur_username = user.username if user else username
        cur_email = user.email if user else ""
        cur_name = farmer.name if farmer else ""
        cur_phone = farmer.phone if farmer else ""
        cur_location = farmer.location if farmer else ""

        with st.form("profile_edit_form"):
            new_username = st.text_input(t("username"), value=cur_username)
            new_email = st.text_input(t("email"), value=cur_email or "")
            if farmer:
                new_name = st.text_input(t("full_name"), value=cur_name)
                new_phone = st.text_input(t("phone"), value=cur_phone)
                new_location = st.text_input(t("address_location"), value=cur_location)
            else:
                new_name = new_phone = new_location = None
            new_password = st.text_input(t("new_password_keep"), type="password")
            submitted = st.form_submit_button(t("save_changes"), type="primary", width="stretch")

        if submitted:
            changes = []
            if user:
                updated_user = User.update_profile(
                    db, user_id,
                    username=new_username if new_username != cur_username else None,
                    email=new_email if new_email != (cur_email or "") else None,
                    password=new_password if new_password else None,
                )
                if updated_user:
                    st.session_state.username = updated_user.username
                    changes.append(t("user_profile_updated"))
            if farmer:
                updated_farmer = Farmer.update_profile(
                    db, farmer.farmer_id,
                    name=new_name if new_name != cur_name else None,
                    location=new_location if new_location != cur_location else None,
                    password=new_password if new_password else None,
                )
                if updated_farmer:
                    st.session_state.farmer_name = updated_farmer.name
                    changes.append(t("farmer_profile_updated"))
            if changes:
                st.success(t("profile_saved"))
                st.rerun()
            else:
                st.info(t("no_changes"))
    except Exception as exc:
        st.warning(t("could_not_load_profile", error=exc))
    finally:
        db.close()


# ── Main Application ──────────────────────────────────────────────────────────

def main():
    """Main application entry."""
    # Install asyncio exception handler to suppress benign ConnectionResetError
    _install_asyncio_exception_handler()

    init_session()
    inject_theme()

    # ── Login gate ──────────────────────────────────────────────────────
    if not st.session_state.logged_in:
        page_login()

    # Ensure auth tables exist
    init_auth_tables()

    # ── Authenticated UI ────────────────────────────────────────────────
    # Auto-start background scheduler (config-driven interval from AUTO_REFRESH_ENABLED/INTERVAL)
    from services import scheduler_service

    # Register atexit handler to stop scheduler cleanly before process exit
    atexit.register(scheduler_service.stop)

    scheduler_service.start()

    render_sidebar_branding()
    render_sidebar_status(st.session_state.db_available)
    render_auto_refresh_status()

    st.sidebar.markdown("##### 🧭 " + t("navigation"))
    render_language_selector()

    # Build navigation based on role (canonical IDs; labels via _nav_label)
    if is_admin():
        nav_ids = [
            "home",
            "admin_dashboard",
            "live_market",
            "predictions",
            "demand",
            "risk",
            "trend",
            "recommendations",
            "analytics",
            "farmers",
            "reports",
            "history",
            "profile",
        ]
    else:
        nav_ids = ["home"]
        if is_farmer():
            nav_ids.append("farmer_dashboard")
        nav_ids += [
            "live_market",
            "predictions",
            "demand",
            "risk",
            "trend",
            "recommendations",
            "history",
            "profile",
        ]

    nav_items = [_nav_label(i) for i in nav_ids]
    nav_labels_to_ids = dict(zip(nav_items, nav_ids))

    # Support programmatic navigation via session state
    nav_to = st.session_state.pop("nav_to", None)
    if nav_to in nav_labels_to_ids.values():
        nav_to = _nav_label(nav_to)
    default_idx = nav_items.index(nav_to) if nav_to in nav_items else 0

    page = st.sidebar.radio(
        t("go_to"),
        nav_items,
        index=default_idx,
        label_visibility="collapsed",
    )
    page_id = nav_labels_to_ids.get(page, "home")

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 " + t("logout")):
        auth_logout()
        st.rerun()

    render_sidebar_footer()

    pages = {
        "home": page_home,
        "farmer_dashboard": page_farmer_dashboard,
        "predictions": page_predictions,
        "recommendations": page_recommendations,
        "analytics": page_analytics,
        "live_market": page_live_market,
        "demand": page_demand_prediction,
        "risk": page_risk_prediction,
        "trend": page_trend_forecast,
        "admin_dashboard": page_admin_dashboard,
        "farmers": page_admin_farmers,
        "reports": page_admin_reports,
        "history": page_prediction_history,
        "profile": page_profile,
    }
    if page_id not in pages:
        page_id = "admin_dashboard" if is_admin() else "home"
    pages[page_id]()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 " + t("run_setup")):
        with st.spinner(t("setup_fetching")):
            try:
                import setup
                setup.bootstrap_live_data()
                setup.train_ml_model()
                setup.train_demand_model()
                setup.train_risk_model()
                st.sidebar.success(t("setup_complete"))
                st.cache_resource.clear()
            except SystemExit:
                st.sidebar.error(t("setup_failed"))


if __name__ == "__main__":
    main()
