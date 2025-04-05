import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from passlib.hash import pbkdf2_sha256
from sqlalchemy import create_engine, text
from collections import deque
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

# ----------------- User Database Configuration -----------------
USER_DB = "user_management.db"

def init_user_db():
    """Initialize SQLite database for user management and create default admin if not exists."""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password_hash TEXT,
                  role TEXT,
                  full_name TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()

    # Check if admin user exists; if not, create default admin.
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    admin = c.fetchone()
    if not admin:
        default_password = "admin123"  # Change this password if needed
        hashed_pw = pbkdf2_sha256.hash(default_password)
        c.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
                  ("admin", hashed_pw, "admin", "System Admin"))
        conn.commit()
        print("Default admin user created: Username: admin, Password: admin123")
    conn.close()

init_user_db()

# ----------------- User Management Functions -----------------
def create_user(username, password, role, full_name):
    """Create new user with hashed password."""
    hashed_pw = pbkdf2_sha256.hash(password)
    try:
        conn = sqlite3.connect(USER_DB)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
                  (username, hashed_pw, role, full_name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
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
    """Verify user credentials."""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and pbkdf2_sha256.verify(password, result[0]):
        return result[1]  # Return role
    return None

# ----------------- Application Database Configuration -----------------
def create_db_engine():
    """Create main database engine."""
    try:
        server = "localhost"  # Ensure correct server name
        database = "R1029"
        driver = "ODBC Driver 17 for SQL Server"
        username = "saa"
        password = "741235689 Asd"
        
        conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};"
        encoded_connection = quote_plus(conn_str)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_connection}", echo=False)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, None
    except SQLAlchemyError as e:
        return None, f"❌ Database connection error: {str(e)}"

# ----------------- Data Fetching -----------------
@st.cache_data(ttl=600)
def fetch_data(query, params=None):
    """Fetch data from main database."""
    engine, error = create_db_engine()
    if error:
        print(error)
        return None
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df
    except SQLAlchemyError as e:
        print(f"❌ Error fetching data: {str(e)}")
        return None

# ----------------- Business Logic Functions -----------------
def calculate_vat(row):
    """Calculate VAT based on currency type."""
    if row['currencyid'] == 2:
        return row['amount'] * 11.18
    elif row['currencyid'] == 3:
        return row['amount'] * 7.45
    return 0.0

def convert_gold(row):
    """Convert gold quantities to 21K."""
    if row['reference'].startswith('S'):
        qty = row.get('qty', np.nan)
        if pd.isna(qty):
            qty = row['amount']
        if row['currencyid'] == 3:  # 21K
            return qty
        elif row['currencyid'] == 2:  # 18K
            return qty * 6 / 7
        elif row['currencyid'] == 14:  # 14K
            return qty * 14 / 21
        elif row['currencyid'] == 4:  # 24K
            return qty * 24 / 21
    else:
        if row['currencyid'] == 2:
            return row['amount'] * 6 / 7
        elif row['currencyid'] == 4:
            return row['amount'] * 24 / 21
    return row['amount']

def process_fifo(debits, credits):
    """Apply FIFO for payment allocation."""
    debits_queue = deque(debits)
    payment_history = []
    for credit in sorted(credits, key=lambda x: x['date']):
        remaining_credit = credit['amount']
        while remaining_credit > 0 and debits_queue:
            current_debit = debits_queue[0]
            amount_to_apply = min(remaining_credit, current_debit['remaining'])
            current_debit['remaining'] -= amount_to_apply
            remaining_credit -= amount_to_apply
            if current_debit['remaining'] <= 0:
                current_debit['paid_date'] = credit['date']
                paid_debit = debits_queue.popleft()
                payment_history.append(paid_debit)
    payment_history.extend([d for d in debits_queue if d['remaining'] > 0])
    return payment_history

def process_report(df, currency_type):
    """Format the aging report."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.floor('D')
    df['paid_date'] = pd.to_datetime(df['paid_date'], errors='coerce').dt.floor('D')
    df['aging_days'] = np.where(
        df['paid_date'].isna(),
        "-",
        (df['paid_date'] - df['date']).dt.days.fillna(0).astype(int)
    )
    num_cols = ['amount', 'remaining', 'vat_amount']
    df[num_cols] = df[num_cols].round(2)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df['paid_date'] = df.apply(
        lambda row: '-' if row['amount'] == 0 else (
            row['paid_date'].strftime('%Y-%m-%d') if not pd.isna(row['paid_date']) else 'Unpaid'
        ),
        axis=1
    )
    suffix = '_gold' if currency_type != 1 else '_cash'
    return df.add_suffix(suffix).rename(columns={
        f'date{suffix}': 'date',
        f'reference{suffix}': 'reference'
    })

def process_transactions(raw_transactions, category_discounts):
    """
    Process raw transactions:
    - For invoices starting with "S" and in cash (currencyid == 1), calculate the final amount after discount.
    - Otherwise, use the original amount.
    """
    if raw_transactions.empty:
        return pd.DataFrame()

    def calc_row(row):
        base_val = row['baseAmount'] + row['basevatamount']
        discount_val = category_discounts.get(row['categoryid'], 0) if pd.notna(row['categoryid']) else 0
        if discount_val != 0:
            return base_val - (discount_val * row['qty'])
        else:
            return base_val

    def process_group(group):
        first_row = group.iloc[0]
        ref = first_row['reference']
        currency = first_row['currencyid']
        original = first_row['amount']
        if ref.startswith('S') and currency == 1:
            valid_satrx = group[~group['baseAmount'].isna()].copy()
            valid_satrx['row_final'] = valid_satrx.apply(calc_row, axis=1)
            final_amount = valid_satrx['row_final'].sum()
        else:
            final_amount = original
        return pd.Series({
            'date': first_row['date'],
            'reference': ref,
            'currencyid': currency,
            'amount': final_amount,
            'original_amount': original
        })

    grouped = raw_transactions.groupby(
        ['functionid', 'recordid', 'date', 'reference', 'currencyid', 'amount']
    )
    transactions = grouped.apply(process_group).reset_index(drop=True)
    transactions['date'] = pd.to_datetime(transactions['date'])
    transactions['converted'] = transactions.apply(convert_gold, axis=1)
    return transactions

def calculate_aging_reports(transactions):
    """Calculate the aging report separating cash and gold columns."""
    cash_debits, cash_credits = [], []
    gold_debits, gold_credits = [], []
    transactions['vat_amount'] = transactions.apply(calculate_vat, axis=1)
    transactions['converted'] = transactions.apply(convert_gold, axis=1)
    for _, row in transactions.iterrows():
        entry = {
            'date': row['date'],
            'reference': row['reference'],
            'amount': abs(row['converted']),
            'remaining': abs(row['converted']),
            'paid_date': None,
            'vat_amount': row['vat_amount']
        }
        if row['currencyid'] == 1:
            if row['amount'] > 0:
                cash_debits.append(entry)
            else:
                cash_credits.append(entry)
        else:
            if row['amount'] > 0:
                gold_debits.append(entry)
            else:
                gold_credits.append(entry)
    cash_results = process_fifo(sorted(cash_debits, key=lambda x: x['date']), cash_credits)
    gold_results = process_fifo(sorted(gold_debits, key=lambda x: x['date']), gold_credits)
    cash_df = process_report(pd.DataFrame(cash_results), 1)
    gold_df = process_report(pd.DataFrame(gold_results), 2)
    merged_df = pd.merge(
        cash_df,
        gold_df,
        on=['date', 'reference'],
        how='outer',
        suffixes=('', '_y')
    ).fillna({
        'amount_cash': 0,
        'remaining_cash': 0,
        'paid_date_cash': 'Unpaid',
        'aging_days_cash': '-',
        'vat_amount_cash': 0,
        'amount_gold': 0,
        'remaining_gold': 0,
        'paid_date_gold': 'Unpaid',
        'aging_days_gold': '-',
        'vat_amount_gold': 0
    })
    final_cols = [
        'date', 'reference',
        'amount_cash', 'remaining_cash', 'paid_date_cash', 'aging_days_cash',
        'amount_gold', 'remaining_gold', 'paid_date_gold', 'aging_days_gold'
    ]
    return merged_df[final_cols]

# ----------------- Authentication Components -----------------
def login_form():
    """Login page with authentication form."""
    st.title("🔐 Invoice Aging System Login")
    
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            user_role = verify_user(username, password)
            if user_role:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = user_role
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

def user_management():
    """User management interface for admins."""
    st.sidebar.header("👥 User Management")
    
    with st.sidebar.expander("➕ Add New User"):
        with st.form("Add User"):
            new_username = st.text_input("Username", key="new_user")
            new_password = st.text_input("Password", type="password", key="new_pass")
            new_role = st.selectbox("Role", ["admin", "user"], key="new_role")
            new_fullname = st.text_input("Full Name", key="new_name")
            if st.form_submit_button("Create User"):
                if create_user(new_username, new_password, new_role, new_fullname):
                    st.success("✅ User created successfully")
                else:
                    st.error("❌ Username already exists")

    with st.sidebar.expander("➖ Remove User"):
        users = get_all_users()
        if users:
            user_list = [f"{user[1]} ({user[3]})" for user in users if user[1] != st.session_state.username]
            selected_user = st.selectbox("Select user to remove", user_list)
            if st.button("Delete User"):
                user_id = [user[0] for user in users if f"{user[1]} ({user[3]})" == selected_user][0]
                delete_user(user_id)
                st.experimental_rerun()
        else:
            st.write("No users to display")

# ----------------- Main Application -----------------
def main_app():
    """Main application interface."""
    st.set_page_config(page_title="Invoice Aging System", layout="wide")
    
    if st.session_state.role == "admin":
        user_management()
    
    with st.sidebar:
        st.write(f"👤 Logged in as: {st.session_state.username} ({st.session_state.role})")
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
    
    st.title("📊 Aging Report")
    
    groups = fetch_data("SELECT recordid, name FROM figrp ORDER BY name")
    if groups is None or groups.empty:
        st.error("❌ No groups found or an error occurred while fetching groups.")
        return

    group_names = ["Select Group..."] + groups['name'].tolist()
    selected_group = st.sidebar.selectbox("Account Group", group_names)
    
    customers = pd.DataFrame()
    if selected_group != "Select Group...":
        group_id = int(groups[groups['name'] == selected_group]['recordid'].values[0])
        customers = fetch_data(
            "SELECT recordid, name, reference FROM fiacc WHERE groupid = :group_id",
            {"group_id": group_id}
        )

    customer_list = ["Select Customer..."] + [f"{row['name']} ({row['reference']})" for _, row in customers.iterrows()]
    selected_customer = st.sidebar.selectbox("Customer Name", customer_list)
    
    start_date = st.sidebar.date_input("Start Date", datetime.now().replace(day=1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    st.sidebar.header("Category Discounts")
    discount_50 = st.sidebar.number_input("احجار عيار 21", min_value=0.0, value=0.0, step=0.01)
    discount_61 = st.sidebar.number_input("سادة عيار 21", min_value=0.0, value=0.0, step=0.01)
    discount_47 = st.sidebar.number_input("ذهب مشغول عيار 18", min_value=0.0, value=0.0, step=0.01)
    discount_62 = st.sidebar.number_input("سادة عيار 18", min_value=0.0, value=0.0, step=0.01)
    discount_48 = st.sidebar.number_input("Estar G18", min_value=0.0, value=0.0, step=0.01)
    discount_45 = st.sidebar.number_input("تعجيل دفع عيار 21", min_value=0.0, value=0.0, step=0.01)
    discount_46 = st.sidebar.number_input("تعجيل دفع عيار 18", min_value=0.0, value=0.0, step=0.01)

    if st.sidebar.button("Generate Report"):
        if selected_customer == "Select Customer...":
            st.error("Please select a customer.")
            return

        customer_id = int(customers.iloc[customer_list.index(selected_customer) - 1]['recordid'])
        query = """
            SELECT 
                f.functionid,
                f.recordid,
                f.date,
                f.reference,
                f.currencyid,
                f.amount,
                s.baseAmount,
                s.baseDiscount,
                s.basevatamount,
                s.qty,
                ivca.recordid as categoryid,
                ivca.parentid as CategoryParent
            FROM fitrx f
            LEFT JOIN satrx s 
                ON f.functionid = s.functionid 
                AND f.recordid = s.recordid
            LEFT JOIN ivit 
                ON s.itemid = ivit.recordid
            LEFT JOIN ivca 
                ON ivit.categoryid = ivca.recordid
            WHERE f.accountid = :acc_id 
        """
        raw_transactions = fetch_data(query, {"acc_id": customer_id})
        
        if raw_transactions is None or raw_transactions.empty:
            st.warning("No transactions found for the given customer ID.")
            return

        category_discounts = {
            50: discount_50,
            47: discount_47,
            61: discount_61,
            62: discount_62,
            48: discount_48
        }
        extra_discount_mapping = {45: discount_45, 46: discount_46}

        # Adjust discounts only for invoices starting with "S" in cash (currencyid == 1)
        s_mask = (raw_transactions['reference'].str.startswith('S')) & (raw_transactions['currencyid'] == 1)
        if s_mask.any():
            s_transactions = raw_transactions[s_mask].copy()
            s_transactions['adjustment_existing'] = np.where(
                s_transactions['categoryid'].isin(category_discounts.keys()),
                s_transactions['baseAmount'] - (s_transactions['categoryid'].map(category_discounts) * s_transactions['qty']),
                s_transactions['baseAmount'] + s_transactions['basevatamount']
            )
            s_transactions['extra_discount'] = np.where(
                s_transactions['CategoryParent'].isin(extra_discount_mapping.keys()),
                s_transactions['CategoryParent'].map(extra_discount_mapping) * s_transactions['qty'],
                0
            )
            s_transactions['adjustment'] = s_transactions['adjustment_existing'] - s_transactions['extra_discount']
            adjustments = s_transactions.groupby(['functionid', 'recordid'])['adjustment'].sum().reset_index()
            raw_transactions = raw_transactions.merge(
                adjustments,
                on=['functionid', 'recordid'],
                how='left',
                suffixes=('', '_adj')
            )
            raw_transactions['adjustment'] = raw_transactions['adjustment'].fillna(0)
            raw_transactions['amount'] = np.where(
                s_mask,
                np.where(
                    raw_transactions['adjustment'] == 0,
                    raw_transactions['baseAmount'] + raw_transactions['basevatamount'],
                    raw_transactions['adjustment']
                ),
                raw_transactions['amount']
            )
        else:
            st.info("No 'S' references with currencyid == 1 found. No discount adjustments applied.")

        transactions = process_transactions(raw_transactions, category_discounts)
        if transactions.empty:
            st.warning("No transactions found.")
            return

        transactions['date'] = pd.to_datetime(transactions['date'])
        final_report = calculate_aging_reports(transactions)
        final_report['date_dt'] = pd.to_datetime(final_report['date'])
        final_report = final_report[
            (final_report['date_dt'] >= pd.to_datetime(start_date)) &
            (final_report['date_dt'] <= pd.to_datetime(end_date))
        ]
        final_report = final_report.drop(columns=['date_dt'])
        final_report = final_report.sort_values(by="date", ascending=True)
        if not final_report.empty:
            st.subheader("Aging Report")
            st.dataframe(final_report, use_container_width=True)
            csv_data = final_report.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="Download Full Report",
                data=csv_data,
                file_name=f"Combined_Aging_Report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No transactions found for the selected date range.")

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        main_app()
    else:
        login_form()
