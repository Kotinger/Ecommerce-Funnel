from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT/"data"
OUT_DIR = DATA/"processed"

# зерно = event; заказ = purchase в одной session
EVENT_COL = "event_type"
DATE_COL = "event_time"
MONEY_COL = "price"
CLIENT_COL = "user_id"
SESSION_COL = "user_session"
PRODUCT_COL = "product_id"
SESSION_KEY = "session_key"
ORDER_KEY = "order_key"
FUNNEL_STEPS = ("view", "cart", "purchase")

def load_data()-> pd.DataFrame:
    files = sorted(DATA.glob("*.csv"))
    parts = []
    for path in files:
        #print("read", path.name)
        part = pd.read_csv(path)
        part.columns = part.columns.str.strip()
        part["source_file"] = path.name
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    #print("shape", df.shape)
    #print("columns", df.columns.tolist())
    #print("isna", df.isna().sum().to_dict())
    #print("event_type", df[EVENT_COL].value_counts().to_dict())
    #print("duplicated", int(df.duplicated().sum()))
    #print("files", [p.name for p in files])
    return df

def check_grain(df: pd.DataFrame)-> str:
    rows = len(df)
    sessions = df[SESSION_COL].nunique(dropna=True)
    users = df[CLIENT_COL].nunique()
    purchases = int((df[EVENT_COL] == "purchase").sum())
    print("rows", rows, "sessions", sessions, "users", users, "-> event")
    print("purchase rows", purchases)
    print("funnel steps", FUNNEL_STEPS)
    if "remove_from_cart" in set(df[EVENT_COL].dropna()):
        print("remove_from_cart")
    return "event"

def prepare_types(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()
    df[CLIENT_COL] = df[CLIENT_COL].astype(str).str.strip()
    df[PRODUCT_COL] = df[PRODUCT_COL].astype(str).str.strip()
    df[SESSION_COL] = df[SESSION_COL].astype(str).str.strip()
    df.loc[df[SESSION_COL].isin(["nan", "None", ""]), SESSION_COL] = pd.NA
    df[MONEY_COL] = pd.to_numeric(df[MONEY_COL], errors="coerce").round(2)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
    df[DATE_COL] = df[DATE_COL].dt.tz_convert("UTC").dt.tz_localize(None)
    df["category_id"] = df["category_id"].astype(str).str.strip()
    df["brand"] = df["brand"].astype(str).str.strip()
    df.loc[df["brand"].isin(["nan", "None", ""]), "brand"] = pd.NA
    df["category_code"] = df["category_code"].astype(str).str.strip()
    df.loc[df["category_code"].isin(["nan", "None", ""]), "category_code"] = pd.NA
    df["category_l1"] = df["category_code"].str.split(".").str[0]
    df["event_hour"] = df[DATE_COL].dt.hour.astype("Int64")
    df["event_date"] = df[DATE_COL].dt.normalize()
    #print("money dtype", df[MONEY_COL].dtype)
    pur = df[df[EVENT_COL] == "purchase"]
    #print("purchase gmv raw", pur[MONEY_COL].sum())
    #print("dates", df[DATE_COL].min(), "->", df[DATE_COL].max())
    #print("NaT", df[DATE_COL].isna().sum())
    return df

def build_clean(df: pd.DataFrame)-> pd.DataFrame:
    clean = df.copy()
    before = len(clean)
    clean = clean.drop_duplicates()
    #print("drop_duplicates", before, "->", len(clean))
    before = len(clean)
    clean = clean.dropna(subset=[SESSION_COL, CLIENT_COL, DATE_COL, EVENT_COL, PRODUCT_COL])
    #print("dropna session/user/time", before, "->", len(clean))
    before = len(clean)
    clean = clean[clean[MONEY_COL].notna() & (clean[MONEY_COL] > 0)]
    #print("price > 0", before, "->", len(clean))
    return clean

def sanity_check(raw: pd.DataFrame, clean: pd.DataFrame)-> None:
    print("--- sanity ---")
    drop_pct = (1 - len(clean) / len(raw)) * 100
    print("rows", len(raw), "->", len(clean))
    print("sessions", raw[SESSION_COL].nunique(dropna=True), "->", clean[SESSION_COL].nunique())
    print("users", raw[CLIENT_COL].nunique(), "->", clean[CLIENT_COL].nunique())
    pur = clean[clean[EVENT_COL] == "purchase"]
    print("purchase gmv", pur[MONEY_COL].sum())
    print("event_type", clean[EVENT_COL].value_counts().to_dict())
    print("brand null%", round(clean["brand"].isna().mean() * 100, 1))
    print("category_code null%", round(clean["category_code"].isna().mean() * 100, 1))

def add_keys(clean: pd.DataFrame)-> pd.DataFrame:
    clean = clean.copy()
    print("--- keys ---")
    clean[SESSION_KEY] = clean[SESSION_COL].astype(str)
    clean[ORDER_KEY] = pd.NA
    is_pur = clean[EVENT_COL] == "purchase"
    clean.loc[is_pur, ORDER_KEY] = (
        clean.loc[is_pur, CLIENT_COL].astype(str) + "|"
        + clean.loc[is_pur, SESSION_COL].astype(str))
    sess_users = clean.groupby(SESSION_KEY)[CLIENT_COL].nunique()
    bad_sess = int((sess_users > 1).sum())
    print("bad session (user >1)", bad_sess)
    pur = clean[is_pur]
    print("order_key", pur[ORDER_KEY].nunique())
    print("session_key", clean[SESSION_KEY].nunique())
    print("purchase rows", len(pur))
    return clean

def build_sessions(clean: pd.DataFrame)-> pd.DataFrame:
    print("--- sessions ---")
    base = clean.dropna(subset=[SESSION_KEY])
    sess = base.groupby(SESSION_KEY, as_index=False).agg(
        user_id=(CLIENT_COL, "first"),
        events=(SESSION_KEY, "size"),
        first_event=(DATE_COL, "min"),
        last_event=(DATE_COL, "max"))
    flags = base.groupby(SESSION_KEY)[EVENT_COL].agg(lambda s: set(s))
    sess["has_view"] = sess[SESSION_KEY].map(lambda s: int("view" in flags.get(s, set())))
    sess["has_cart"] = sess[SESSION_KEY].map(lambda s: int("cart" in flags.get(s, set())))
    sess["has_purchase"] = sess[SESSION_KEY].map(lambda s: int("purchase" in flags.get(s, set())))
    sess["has_remove"] = sess[SESSION_KEY].map(lambda s: int("remove_from_cart" in flags.get(s, set())))
    pur = clean[clean[EVENT_COL] == "purchase"]
    if len(pur):
        buy = pur.groupby(SESSION_KEY, as_index=False).agg(
            gmv=(MONEY_COL, "sum"),
            orders=(ORDER_KEY, "nunique"))
        sess = sess.merge(buy, on=SESSION_KEY, how="left")
    else:
        sess["gmv"] = 0.0
        sess["orders"] = 0
    sess["gmv"] = sess["gmv"].fillna(0.0)
    sess["orders"] = sess["orders"].fillna(0).astype(int)
    print("sessions", len(sess))
    print("view/cart/purchase", int(sess["has_view"].sum()), int(sess["has_cart"].sum()), int(sess["has_purchase"].sum()))
    return sess

def build_people(clean: pd.DataFrame)-> pd.DataFrame:
    print("--- people ---")
    base = clean.dropna(subset=[CLIENT_COL])
    people = base.groupby(CLIENT_COL, as_index=False).agg(
        sessions=(SESSION_KEY, "nunique"),
        events=(SESSION_KEY, "size"),
        first_event=(DATE_COL, "min"),
        last_event=(DATE_COL, "max"))
    pur = clean[clean[EVENT_COL] == "purchase"]
    if len(pur):
        buy = pur.groupby(CLIENT_COL, as_index=False).agg(
            gmv=(MONEY_COL, "sum"),
            orders=(ORDER_KEY, "nunique"),
            first_order=(DATE_COL, "min"))
        people = people.merge(buy, on=CLIENT_COL, how="left")
    else:
        people["gmv"] = 0.0
        people["orders"] = 0
        people["first_order"] = pd.NaT
    people["gmv"] = people["gmv"].fillna(0.0)
    people["orders"] = people["orders"].fillna(0).astype(int)
    flags = base.groupby(CLIENT_COL)[EVENT_COL].agg(lambda s: set(s))
    people["has_view"] = people[CLIENT_COL].map(lambda u: int("view" in flags.get(u, set())))
    people["has_cart"] = people[CLIENT_COL].map(lambda u: int("cart" in flags.get(u, set())))
    people["has_purchase"] = people[CLIENT_COL].map(lambda u: int("purchase" in flags.get(u, set())))
    print("users", len(people))
    print("repeat sessions>=2", int((people["sessions"] >= 2).sum()))
    print("repeat%", round((people["sessions"] >= 2).mean() * 100, 1))
    buyers = people[people["orders"] >= 1]
    print("buyers", len(buyers))
    if len(buyers):
        print("repeat buyers>=2", int((buyers["orders"] >= 2).sum()))
        print("repeat buyers%", round((buyers["orders"] >= 2).mean() * 100, 1))
        print("ltv min/median/max", buyers["gmv"].min(), buyers["gmv"].median(), buyers["gmv"].max())
    print("gmv people", people["gmv"].sum())
    return people

def save_tables(clean: pd.DataFrame, people: pd.DataFrame, sessions: pd.DataFrame)-> None:
    print("--- save ---")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUT_DIR/"clean.parquet", index=False)
    people.to_parquet(OUT_DIR/"people.parquet", index=False)
    sessions.to_parquet(OUT_DIR/"sessions.parquet", index=False)
    back = pd.read_parquet(OUT_DIR/"clean.parquet")
    print("saved clean", len(back))
    print("saved people", len(people))
    print("saved sessions", len(sessions))

def main()-> None:
    raw = load_data()
    check_grain(raw)
    typed = prepare_types(raw)
    clean = build_clean(typed)
    sanity_check(raw, clean)
    clean = add_keys(clean)
    sessions = build_sessions(clean)
    people = build_people(clean)
    save_tables(clean, people, sessions)

if __name__ == "__main__":
    main()
