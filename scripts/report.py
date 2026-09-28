from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT/"data"/"processed"

EVENT_COL = "event_type"
DATE_COL = "event_time"
MONEY_COL = "price"
CLIENT_COL = "user_id"
SESSION_KEY = "session_key"
ORDER_KEY = "order_key"
PRODUCT_COL = "product_id"
BRAND_COL = "brand"
CAT_COL = "category_code"
CAT_L1 = "category_l1"

def load_tables():
    clean = pd.read_parquet(OUT_DIR/"clean.parquet")
    people = pd.read_parquet(OUT_DIR/"people.parquet")
    sessions = pd.read_parquet(OUT_DIR/"sessions.parquet")
    return clean, people, sessions

def purchases(clean: pd.DataFrame)-> pd.DataFrame:
    return clean[clean[EVENT_COL] == "purchase"].copy()

def kpi_totals(clean: pd.DataFrame)-> None:
    pur = purchases(clean)
    gmv = pur[MONEY_COL].sum()
    orders = pur[ORDER_KEY].nunique()
    aov = gmv / orders if orders else 0
    print("events", len(clean))
    print("sessions", clean[SESSION_KEY].nunique())
    print("users", clean[CLIENT_COL].nunique())
    print("gmv", gmv, "orders", orders, "aov", aov)
    print("purchase rows", len(pur))
    print("purchase rows / order", round(len(pur) / orders, 2) if orders else 0)
    print("period", clean[DATE_COL].min(), "->", clean[DATE_COL].max())
    print("event_type", clean[EVENT_COL].value_counts().to_dict())

def kpi_funnel(sessions: pd.DataFrame, people: pd.DataFrame)-> None:
    print("--- funnel sessions ---")
    sv = int(sessions["has_view"].sum())
    sc = int(sessions["has_cart"].sum())
    sp = int(sessions["has_purchase"].sum())
    print("sessions view/cart/purchase", sv, sc, sp)
    print("view->cart%", round(sc / sv * 100, 2) if sv else 0)
    print("cart->purchase%", round(sp / sc * 100, 2) if sc else 0)
    print("view->purchase%", round(sp / sv * 100, 2) if sv else 0)
    print("drop view-cart", sv - sc, "drop cart-pur", sc - sp)
    no_view = int(((sessions["has_purchase"] == 1) & (sessions["has_view"] == 0)).sum())
    no_cart = int(((sessions["has_purchase"] == 1) & (sessions["has_cart"] == 0)).sum())
    print("purchase sess without view", no_view, "without cart", no_cart)
    print("--- funnel users ---")
    uv = int(people["has_view"].sum())
    uc = int(people["has_cart"].sum())
    up = int(people["has_purchase"].sum())
    print("users view/cart/purchase", uv, uc, up)
    print("view->cart%", round(uc / uv * 100, 2) if uv else 0)
    print("cart->purchase%", round(up / uc * 100, 2) if uc else 0)
    print("view->purchase%", round(up / uv * 100, 2) if uv else 0)

def kpi_year_month(clean: pd.DataFrame)-> pd.DataFrame:
    pur = purchases(clean)
    y_m = pur.copy()
    y_m["year"] = y_m[DATE_COL].dt.year
    y_m["month"] = y_m[DATE_COL].dt.month
    out = y_m.groupby(["year", "month"], as_index=False).agg(
        gmv=(MONEY_COL, "sum"),
        orders=(ORDER_KEY, "nunique"),
        purchase_rows=(ORDER_KEY, "size"))
    out["aov"] = out["gmv"] / out["orders"]
    out = out.sort_values(["year", "month"])
    print(out)
    return out

def kpi_hour(clean: pd.DataFrame)-> pd.DataFrame:
    pur = purchases(clean)
    out = pur.groupby("event_hour", as_index=False).agg(
        gmv=(MONEY_COL, "sum"),
        orders=(ORDER_KEY, "nunique"))
    out["aov"] = out["gmv"] / out["orders"]
    out = out.sort_values("event_hour")
    print(out)
    print("peak hours")
    print(out.sort_values("gmv", ascending=False).head(3))
    return out

def kpi_repeat(people: pd.DataFrame)-> None:
    print("users", len(people))
    print("repeat sessions>=2", int((people["sessions"] >= 2).sum()))
    print("repeat sessions%", round((people["sessions"] >= 2).mean() * 100, 1))
    buyers = people[people["orders"] >= 1]
    print("buyers", len(buyers))
    if len(buyers):
        print("repeat buyers>=2", int((buyers["orders"] >= 2).sum()))
        print("repeat buyers%", round((buyers["orders"] >= 2).mean() * 100, 1))
        print("ltv min/median/max", buyers["gmv"].min(), buyers["gmv"].median(), buyers["gmv"].max())
    print("gmv people", people["gmv"].sum())

def kpi_category(clean: pd.DataFrame)-> None:
    pur = purchases(clean)
    print("sku purchase", pur[PRODUCT_COL].nunique())
    by_l1 = pur.groupby(CAT_L1, as_index=False, dropna=False).agg(
        gmv=(MONEY_COL, "sum"),
        orders=(ORDER_KEY, "nunique"),
        rows=(ORDER_KEY, "size"))
    by_l1["aov"] = by_l1["gmv"] / by_l1["orders"]
    print("--- category_l1 ---")
    print(by_l1.sort_values("gmv", ascending=False).head(12))
    by_brand = pur.groupby(BRAND_COL, as_index=False, dropna=False).agg(
        gmv=(MONEY_COL, "sum"),
        orders=(ORDER_KEY, "nunique"))
    print("--- brand top ---")
    print(by_brand.sort_values("gmv", ascending=False).head(10))

def kpi_conversion_by_l1(clean: pd.DataFrame)-> None:
    print("--- conversion by category_l1 (sessions) ---")
    rows = []
    for cat, part in clean.dropna(subset=[CAT_L1]).groupby(CAT_L1):
        sess = part.groupby(SESSION_KEY)[EVENT_COL].agg(lambda s: set(s))
        sv = sum("view" in s for s in sess)
        sc = sum("cart" in s for s in sess)
        sp = sum("purchase" in s for s in sess)
        if sv < 1000:
            continue
        rows.append({
            "category_l1": cat,
            "view_sess": sv,
            "cart_sess": sc,
            "pur_sess": sp,
            "view_pur_pct": round(sp / sv * 100, 2) if sv else 0})
    out = pd.DataFrame(rows).sort_values("view_pur_pct", ascending=False)
    print(out.head(12))

def main()-> None:
    clean, people, sessions = load_tables()
    kpi_totals(clean)
    kpi_funnel(sessions, people)
    kpi_year_month(clean)
    kpi_hour(clean)
    kpi_repeat(people)
    kpi_category(clean)
    kpi_conversion_by_l1(clean)

if __name__ == "__main__":
    main()
