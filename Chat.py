import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
import sqlite3
from datetime import datetime
from passlib.hash import pbkdf2_sha256
from sqlalchemy import create_engine, text
from collections import deque
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

# ----------------- Helper: Rerun Function -----------------
def rerun():
    """Call experimental_rerun if available, otherwise prompt user to refresh."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.write("Please refresh the page manually.")

# ----------------- User Database Configuration -----------------
USER_DB = "user_management.db"

def init_user_db():
    """Initialize SQLite database for user management, create default admin, and add force flag."""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    # Create users table with force_password_change flag
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password_hash TEXT,
                  role TEXT,
                  full_name TEXT,
                  force_password_change INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()

    # Ensure force_password_change column exists (for migrations)
    c.execute("PRAGMA table_info(users)")
    cols = [row[1] for row in c.fetchall()]
    if 'force_password_change' not in cols:
        c.execute("ALTER TABLE users ADD COLUMN force_password_change INTEGER DEFAULT 0")
        conn.commit()

    # Create default admin with forced password change
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    admin = c.fetchone()
    if not admin:
        default_password = "admin123"
        hashed_pw = pbkdf2_sha256.hash(default_password)
        c.execute(
            "INSERT INTO users (username, password_hash, role, full_name, force_password_change) VALUES (?, ?, ?, ?, 1)",
            ("admin", hashed_pw, "admin", "System Admin")
        )
        conn.commit()
        print("Default admin user created: Username: admin, Password: admin123 (force password change)")
    conn.close()

init_user_db()

# ----------------- User Management Functions -----------------
def create_user(username, password, role, full_name):
    """Create new user with hashed password and force password change."""
    hashed_pw = pbkdf2_sha256.hash(password)
    try:
        conn = sqlite3.connect(USER_DB)
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, password_hash, role, full_name, force_password_change) VALUES (?, ?, ?, ?, 1)",
            (username, hashed_pw, role, full_name)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def reset_user_password(user_id, new_password):
    """Reset user's password and require change on next login."""
    hashed_pw = pbkdf2_sha256.hash(new_password)
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET password_hash = ?, force_password_change = 1 WHERE id = ?",
        (hashed_pw, user_id)
    )
    conn.commit()
    conn.close()

def delete_user(user_id):
    """Delete user by ID."""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_all_users():
    """Get all users."""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute("SELECT id, username, role, full_name FROM users")
    users = c.fetchall()
    conn.close()
    return users

def verify_user(username, password):
    """Verify user credentials and return role and force flag."""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute(
        "SELECT id, password_hash, role, force_password_change FROM users WHERE username = ?", 
        (username,)
    )
    result = c.fetchone()
    conn.close()
    if result and pbkdf2_sha256.verify(password, result[1]):
        user_id, _, role, force_flag = result
        return user_id, role, bool(force_flag)
    return None, None, False

# ----------------- Password Change -----------------
def change_password(user_id, new_password):
    """Update password and clear force flag."""
    hashed_pw = pbkdf2_sha256.hash(new_password)
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET password_hash = ?, force_password_change = 0 WHERE id = ?",
        (hashed_pw, user_id)
    )
    conn.commit()
    conn.close()

# ----------------- Application Database Configuration -----------------
def create_db_engine():
    """Create main database engine."""
    try:
        server = "localhost"
        database = "R1029"
        driver = "ODBC Driver 17 for SQL Server"
        username = "saa"
        password = "741235689 Asd"
        conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};"
        encoded = quote_plus(conn_str)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={encoded}", echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, None
    except SQLAlchemyError as e:
        return None, f"❌ Database connection error: {str(e)}"

# ----------------- Data Fetching -----------------
@st.cache_data(ttl=600)
def fetch_data(query, params=None):
    engine, error = create_db_engine()
    if error:
        print(error)
        return None
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    except SQLAlchemyError as e:
        print(f"❌ Error fetching data: {e}")
        return None

# ----------------- Business Logic Functions -----------------
def calculate_vat(row):
    if row['currencyid'] == 2:
        return row['amount'] * 11.18
    elif row['currencyid'] == 3:
        return row['amount'] * 7.45
    return 0.0

def convert_gold(row):
    # ... (same as before) ...
    if row['reference'].startswith('S'):
        qty = row.get('qty', np.nan)
        if pd.isna(qty): qty = row['amount']
        if row['currencyid'] == 3:
            return qty
        elif row['currencyid'] == 2:
            return qty * 6 / 7
        elif row['currencyid'] == 14:
            return qty * 14 / 21
        elif row['currencyid'] == 4:
            return qty * 24 / 21
    else:
        if row['currencyid'] == 2:
            return row['amount'] * 6 / 7
        elif row['currencyid'] == 4:
            return row['amount'] * 24 / 21
    return row['amount']

def process_fifo(debits, credits):
    debits_q = deque(debits)
    history = []
    for credit in sorted(credits, key=lambda x: x['date']):
        rem = credit['amount']
        while rem > 0 and debits_q:
            d = debits_q[0]
            apply_amt = min(rem, d['remaining'])
            d['remaining'] -= apply_amt
            rem -= apply_amt
            if d['remaining'] <= 0:
                d['paid_date'] = credit['date']
                history.append(debits_q.popleft())
    history.extend([d for d in debits_q if d['remaining'] > 0])
    return history

def process_report(df, currency_type):
    df['date'] = pd.to_datetime(df['date']).dt.floor('D')
    df['paid_date'] = pd.to_datetime(df['paid_date']).dt.floor('D')
    df['aging_days'] = np.where(df['paid_date'].isna(), '-', (df['paid_date'] - df['date']).dt.days)
    for col in ['amount','remaining','vat_amount']:
        df[col] = df[col].round(2)
    df['paid_date'] = df.apply(lambda r: r['paid_date'].strftime('%Y-%m-%d') if pd.notna(r['paid_date']) else 'Unpaid', axis=1)
    suffix = '_gold' if currency_type != 1 else '_cash'
    return df.rename(columns={
        'date': 'date',
        'reference': 'reference'
    }).add_suffix(suffix).rename(columns={f'date{suffix}':'date',f'reference{suffix}':'reference'})

def process_transactions(raw, discounts):
    # ... same as before ...
    if raw.empty: return pd.DataFrame()
    def calc_row(r):
        base = r['baseAmount'] + r['basevatamount']
        disc = discounts.get(r['categoryid'], 0)
        return base - (disc * r['qty']) if disc else base
    def group_fn(g):
        fr = g.iloc[0]
        ref, cur, orig = fr['reference'], fr['currencyid'], fr['amount']
        if ref.startswith('S') and cur == 1:
            valid = g[~g['baseAmount'].isna()].copy()
            valid['final'] = valid.apply(calc_row, axis=1)
            amt = valid['final'].sum()
        else:
            amt = orig
        return pd.Series({'date': fr['date'], 'reference': ref, 'currencyid': cur, 'amount': amt, 'original_amount': orig})
    grp = raw.groupby(['functionid','recordid','date','reference','currencyid','amount'])
    txs = grp.apply(group_fn).reset_index(drop=True)
    txs['date'] = pd.to_datetime(txs['date'])
    txs['converted'] = txs.apply(convert_gold, axis=1)
    return txs

def calculate_aging_reports(transactions):
    cash_debits, cash_credits, gold_debits, gold_credits = [],[],[],[]
    transactions['vat_amount'] = transactions.apply(calculate_vat, axis=1)
    transactions['converted'] = transactions.apply(convert_gold, axis=1)
    for _, r in transactions.iterrows():
        entry = {'date': r['date'], 'reference': r['reference'], 'amount': abs(r['converted']), 'remaining': abs(r['converted']), 'paid_date': None, 'vat_amount': r['vat_amount']}
        if r['currencyid'] == 1:
            (cash_debits if r['amount']>0 else cash_credits).append(entry)
        else:
            (gold_debits if r['amount']>0 else gold_credits).append(entry)
    cash = process_fifo(sorted(cash_debits, key=lambda x: x['date']), cash_credits)
    gold = process_fifo(sorted(gold_debits, key=lambda x: x['date']), gold_credits)
    cash_df = process_report(pd.DataFrame(cash),1)
    gold_df = process_report(pd.DataFrame(gold),2)
    df = pd.merge(cash_df, gold_df, on=['date','reference'], how='outer').fillna({
        'amount_cash':0,'remaining_cash':0,'paid_date_cash':'Unpaid','aging_days_cash':'-','vat_amount_cash':0,
        'amount_gold':0,'remaining_gold':0,'paid_date_gold':'Unpaid','aging_days_gold':'-','vat_amount_gold':0
    })
    return df[['date','reference','amount_cash','remaining_cash','paid_date_cash','aging_days_cash','amount_gold','remaining_gold','paid_date_gold','aging_days_gold']]

# ----------------- Authentication Components -----------------
def login_form():
    st.title("🔐 Invoice Aging System Login")
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            uid, role, force = verify_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.user_id = uid
                st.session_state.username = username
                st.session_state.role = role
                st.session_state.force_password_change = force
                rerun()
            else:
                st.error("Invalid username or password")

def password_change_form():
    st.title("🔑 Change Your Password")
    st.write("You must change your password before continuing.")
    with st.form("ChangePassword"):
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Update Password"):
            if not new_pw or new_pw != confirm_pw:
                st.error("Passwords do not match or are empty.")
            else:
                change_password(st.session_state.user_id, new_pw)
                st.success("Password updated! Please log in again.")
                for k in list(st.session_state.keys()): del st.session_state[k]
                rerun()

# ----------------- User Management Interface -----------------
def user_management():
    st.sidebar.header("👥 User Management")
    # Add New User
    with st.sidebar.expander("➕ Add New User"):
        with st.form("Add User"):
            new_username = st.text_input("Username", key="new_user")
            new_password = st.text_input("Password", type="password", key="new_pass")
            new_role = st.selectbox("Role", ["admin", "user"], key="new_role")
            new_fullname = st.text_input("Full Name", key="new_name")
            if st.form_submit_button("Create User"):
                if create_user(new_username, new_password, new_role, new_fullname):
                    st.success("✅ User created successfully. They will be prompted to change password on first login.")
                else:
                    st.error("❌ Username already exists")
    # Reset Password
    with st.sidebar.expander("🔄 Reset User Password"):
        users = get_all_users()
        options = [f"{u[1]} ({u[3]})" for u in users]
        selected = st.selectbox("Select user", options, key="reset_user")
        new_pw = st.text_input("New Password", type="password", key="reset_pw")
        if st.button("Reset Password"):
            uid = [u[0] for u in users if f"{u[1]} ({u[3]})" == selected][0]
            if new_pw:
                reset_user_password(uid, new_pw)
                st.success("✅ Password reset. User must change password at next login.")
            else:
                st.error("Enter a new password to reset.")
    # Remove User
    with st.sidebar.expander("➖ Remove User"):
        users = get_all_users()
        if users:
            user_list = [f"{u[1]} ({u[3]})" for u in users if u[1] != st.session_state.username]
            selected_user = st.selectbox("Select user to remove", user_list, key="del_user")
            if st.button("Delete User"):
                user_id = [u[0] for u in users if f"{u[1]} ({u[3]})" == selected_user][0]
                delete_user(user_id)
                rerun()
        else:
            st.write("No users to display")

# ----------------- Main Application -----------------
def main_app():
    if st.session_state.get('force_password_change', False):
        password_change_form()
        return
    st.set_page_config(page_title="Invoice Aging System", layout="wide")
    if st.session_state.role == "admin":
        user_management()

    with st.sidebar:
        st.write(f"👤 Logged in as: {st.session_state.username} ({st.session_state.role})")
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            rerun()

    st.title("📊 Aging Report")
    # -- Group and Customer Selection --
    groups = fetch_data("SELECT recordid, name FROM figrp ORDER BY name")
    if groups is None or groups.empty:
        st.error("❌ No groups found or an error occurred while fetching groups.")
        return
    group_names = ["Select Group..."] + groups['name'].tolist()
    selected_group = st.sidebar.selectbox("Account Group", group_names)
    customers = pd.DataFrame()
    if selected_group != "Select Group...":
        gid = int(groups[groups['name']==selected_group]['recordid'].values[0])
        customers = fetch_data(
            "SELECT recordid, name, reference FROM fiacc WHERE groupid = :g", {"g": gid}
        )
    cust_list = ["Select Customer..."] + [f"{r['name']} ({r['reference']})" for _,r in customers.iterrows()]
    selected_customer = st.sidebar.selectbox("Customer Name", cust_list)
    start_date = st.sidebar.date_input("Start Date", datetime.now().replace(day=1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    # -- Discounts --
    st.sidebar.header("Category Discounts")
    discount_50 = st.sidebar.number_input("احجار عيار 21", 0.0, 1000.0, 0.0)
    discount_61 = st.sidebar.number_input("سادة عيار 21", 0.0, 1000.0, 0.0)
    discount_47 = st.sidebar.number_input("ذهب مشغول عيار 18", 0.0, 1000.0, 0.0)
    discount_62 = st.sidebar.number_input("سادة عيار 18", 0.0, 1000.0, 0.0)
    discount_48 = st.sidebar.number_input("Estar G18", 0.0, 1000.0, 0.0)
    discount_45 = st.sidebar.number_input("تعجيل دفع عيار 21", 0.0, 1000.0, 0.0)
    discount_46 = st.sidebar.number_input("تعجيل دفع عيار 18", 0.0, 1000.0, 0.0)
    if st.sidebar.button("Generate Report"):
        if selected_customer == "Select Customer...":
            st.error("Please select a customer.")
            return
        cid = int(customers.iloc[cust_list.index(selected_customer)-1]['recordid'])
        query = """
            SELECT f.functionid,f.recordid,f.date,f.reference,f.currencyid,f.amount,
                   s.baseAmount,s.baseDiscount,s.basevatamount,s.qty,
                   ivca.recordid as categoryid, ivca.parentid as CategoryParent
            FROM fitrx f
            LEFT JOIN satrx s ON f.functionid=s.functionid AND f.recordid=s.recordid
            LEFT JOIN ivit ON s.itemid=ivit.recordid
            LEFT JOIN ivca ON ivit.categoryid=ivca.recordid
            WHERE f.accountid=:acc
        """
        raw = fetch_data(query, {"acc": cid})
        if raw is None or raw.empty:
            st.warning("No transactions found for the given customer ID.")
            return
        discounts = {50: discount_50,47:discount_47,61:discount_61,62:discount_62,48:discount_48}
        extras = {45: discount_45,46:discount_46}
        # Adjust for 'S' invoices
        mask = raw['reference'].str.startswith('S') & (raw['currencyid']==1)
        if mask.any():
            sraw = raw[mask].copy()
            sraw['adj_exist'] = np.where(
                sraw['categoryid'].isin(discounts),
                sraw['baseAmount'] - (sraw['categoryid'].map(discounts)*sraw['qty']),
                sraw['baseAmount']+sraw['basevatamount']
            )
            sraw['extra_disc'] = np.where(
                sraw['CategoryParent'].isin(extras),
                sraw['CategoryParent'].map(extras)*sraw['qty'],0
            )
            sraw['adjustment'] = sraw['adj_exist'] - sraw['extra_disc']
            adjs = sraw.groupby(['functionid','recordid'])['adjustment'].sum().reset_index()
            raw = raw.merge(adjs,on=['functionid','recordid'],how='left')
            raw['adjustment'] = raw['adjustment'].fillna(0)
            raw['amount'] = np.where(mask, np.where(raw['adjustment']==0, raw['baseAmount']+raw['basevatamount'], raw['adjustment']), raw['amount'])
        else:
            st.info("No 'S' cash invoices found. No adjustments applied.")
        txs = process_transactions(raw, discounts)
        if txs.empty:
            st.warning("No transactions to process.")
            return
        report = calculate_aging_reports(txs)
        report['date_dt'] = pd.to_datetime(report['date'])
        report = report[(report['date_dt']>=pd.to_datetime(start_date))&(report['date_dt']<=pd.to_datetime(end_date))]
        report = report.drop(columns=['date_dt']).sort_values('date')
        if not report.empty:
            st.subheader("Aging Report")
            st.dataframe(report, use_container_width=True)
            csv = report.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("Download Full Report", csv, file_name=f"Aging_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            if st.button("Print Report"):
                components.html("<script>window.print();</script>", height=0)
        else:
            st.warning("No data for selected date range.")

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        main_app()
    else:
        login_form()
