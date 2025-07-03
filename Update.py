import streamlit as st
import pandas as pd
import numpy as np
import sqlitecloud
from datetime import datetime, timedelta
from passlib.hash import pbkdf2_sha256
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from collections import deque
from urllib.parse import quote_plus
import time
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

# -------------- Helpers --------------
def rerun():
    st.experimental_rerun()

# -------------- App-DB Setup -------------
def create_db_engine():
    """إنشاء محرك اتصال بقاعدة البيانات."""
    try:
        server = "52.48.117.197"
        database = "R1029"
        username = "sa"
        password = "Argus@NEG"
        driver = "ODBC Driver 17 for SQL Server"
        connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=Yes;Connection Timeout=30"
        encoded_connection = quote_plus(connection_string)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_connection}")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, None
    except Exception as e:
        return None, str(e)

# ----------------- Data Fetching -----------------
@st.cache_data(ttl=600)
def fetch_data(query, params=None):
    engine, error = create_db_engine()
    if error:
        st.error(f"❌ Database connection failed: {error}")
        return None
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    except SQLAlchemyError as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

def calculate_vat(row):
    if row['currencyid'] == 2:
        return row['amount'] * 11.18
    elif row['currencyid'] == 3:
        return row['amount'] * 7.45
    return 0.0

def convert_gold(row):
    if row['reference'].startswith('S'):
        qty = row.get('qty', np.nan)
        if pd.isna(qty):
            qty = row['amount']
        if row['currencyid'] == 3:
            result = qty
        elif row['currencyid'] == 2:
            result = qty * 6 / 7
        elif row['currencyid'] == 14:
            result = qty * 14 / 21
        elif row['currencyid'] == 4:
            result = qty * 24 / 21
        else:
            result = row['amount']
    else:
        if row['currencyid'] == 2:
            result = row['amount'] * 6 / 7
        elif row['currencyid'] == 4:
            result = row['amount'] * 24 / 21
        else:
            result = row['amount']
    return round(result, 2)

# ----------------- FIFO Processing -----------------
def process_fifo(debits, credits):
    """Process transactions using FIFO with priority for functionid=3104"""
    # Separate priority transactions (functionid=3104)
    priority_debits = [d for d in debits if d.get('is_priority', False)]
    normal_debits = [d for d in debits if not d.get('is_priority', False)]
    
    # Sort both lists by date
    priority_debits = sorted(priority_debits, key=lambda x: x['date'])
    normal_debits = sorted(normal_debits, key=lambda x: x['date'])
    
    # Combine with priority first
    debits_q = deque(priority_debits + normal_debits)
    
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
    history.extend(debits_q)
    
    return history

def process_report(df, currency_type):
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.floor('D')
    df['paid_date'] = pd.to_datetime(df['paid_date'], errors='coerce').dt.floor('D')
    df['aging_days'] = np.where(df['paid_date'].isna(), '-',
                                (df['paid_date'] - df['date']).dt.days.fillna(0).astype(int))
    for col in ['amount', 'remaining', 'vat_amount']:
        df[col] = df[col].round(2)
    df['paid_date'] = df.apply(lambda r: r['paid_date'].strftime('%Y-%m-%d') if pd.notna(r['paid_date']) else 'Unpaid',
                               axis=1)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    suffix = '_gold' if currency_type != 1 else '_cash'
    return df.rename(columns={'date': 'date', 'reference': 'reference'}).add_suffix(suffix).rename(
        columns={f'date{suffix}': 'date', f'reference{suffix}': 'reference'})

def process_transactions(raw, discounts, extras, start_date):
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    raw = raw.dropna(subset=['date'])
    
    def calc_row(r):
        base = r['baseAmount'] + r['basevatamount']
        if pd.to_datetime(r['date']) >= start_date:
            disc = discounts.get(r['categoryid'], 0)
            extra = extras.get(r['categoryid'], 0)
            return base - (disc * r['qty']) - (extra * r['qty'])
        return base

    def group_fn(g):
        fr = g.iloc[0]
        ref, cur, orig = fr['reference'], fr['currencyid'], fr['amount']
        if ref.startswith('S') and cur == 1:
            valid = g[~g['baseAmount'].isna()].copy()
            valid['final'] = valid.apply(calc_row, axis=1)
            amt = valid['final'].sum()
        else:
            amt = orig
        return pd.Series({'date': fr['date'], 'reference': ref,
                          'currencyid': cur, 'amount': amt, 'original_amount': orig,
                          'functionid': fr['functionid']})

    grp = raw.groupby(['functionid', 'recordid', 'date', 'reference', 'currencyid', 'amount'])
    txs = grp.apply(group_fn).reset_index(drop=True)
    txs['date'] = pd.to_datetime(txs['date'])
    txs['converted'] = txs.apply(convert_gold, axis=1)
    
    # Mark priority transactions (functionid=3104)
    txs['is_priority'] = txs['functionid'] == 3104
    
    return txs

def calculate_aging_reports(transactions):
    """حساب تقرير Aging المُجمّع باستخدام FIFO مع استثناء المعاملات ذات الأولوية"""
    cash_debits, cash_credits, gold_debits, gold_credits = [], [], [], []
    transactions['vat_amount'] = transactions.apply(calculate_vat, axis=1)
    transactions['converted'] = transactions.apply(convert_gold, axis=1)
    
    for _, r in transactions.iterrows():
        # Skip priority transactions in aggregated report
        if r['is_priority']:
            continue
            
        entry = {'date': r['date'], 'reference': r['reference'],
                 'amount': abs(r['converted']), 'remaining': abs(r['converted']),
                 'paid_date': None, 'vat_amount': r['vat_amount'],
                 'is_priority': r['is_priority']}
                 
        if r['currencyid'] == 1:
            (cash_debits if r['amount'] > 0 else cash_credits).append(entry)
        else:
            (gold_debits if r['amount'] > 0 else gold_credits).append(entry)
            
    cash = process_fifo(sorted(cash_debits, key=lambda x: x['date']), cash_credits)
    gold = process_fifo(sorted(gold_debits, key=lambda x: x['date']), gold_credits)
    
    cash_df = process_report(pd.DataFrame(cash), 1)
    gold_df = process_report(pd.DataFrame(gold), 2)
    
    df = pd.merge(cash_df, gold_df, on=['date', 'reference'], how='outer').fillna({
        'amount_gold': 0, 'remaining_gold': 0, 'paid_date_gold': '-', 'aging_days_gold': '-', 'vat_amount_gold': 0,
        'amount_cash': 0, 'remaining_cash': 0, 'paid_date_cash': '-', 'aging_days_cash': '-', 'vat_amount_cash': 0,
    })
    
    return df[['date', 'reference', 'amount_gold', 'remaining_gold', 'paid_date_gold', 'aging_days_gold',
               'amount_cash', 'remaining_cash', 'paid_date_cash', 'aging_days_cash']]

# ----------------- Detailed FIFO Processing -----------------
def process_fifo_detailed(debits, credits):
    """
    Simulate FIFO with high performance using integer arithmetic (cents).
    Includes priority transactions and tracks credit reference.
    """
    cutoff = pd.to_datetime("2023-01-01")
    
    # Preprocess debits: filter, round amounts, and convert to cents
    debits_processed = []
    for d in debits:
        if d['date'] < cutoff:
            continue
        inv_amt = round(d['amount'], 2)
        debits_processed.append({
            'date': d['date'],
            'reference': d['reference'],
            'currencyid': d['currencyid'],
            'invoice_amount': inv_amt,
            'remaining_cents': int(inv_amt * 100),
            'is_priority': d.get('is_priority', False)
        })
    
    # Separate priority and normal debits
    priority_debits = [d for d in debits_processed if d['is_priority']]
    normal_debits = [d for d in debits_processed if not d['is_priority']]
    
    # Sort each group by date
    priority_debits = sorted(priority_debits, key=lambda x: x['date'])
    normal_debits = sorted(normal_debits, key=lambda x: x['date'])
    
    # Combine with priority first
    all_debits = priority_debits + normal_debits
    debits_q = deque(all_debits)
    
    # Preprocess credits: filter, round amounts, and convert to cents
    sorted_credits = sorted([
        {
            'date': c['date'],
            'amount_cents': int(round(c['amount'], 2) * 100),
            'reference': c.get('reference', 'Unknown-Credit')
        }
        for c in credits if c['date'] >= cutoff
    ], key=lambda x: x['date'])
    
    detailed = []
    today = pd.Timestamp(datetime.now().date())
    
    # Process credits in chronological order
    for credit in sorted_credits:
        rem_credit_cents = credit['amount_cents']
        while rem_credit_cents > 0 and debits_q:
            d = debits_q[0]
            if d['remaining_cents'] <= 0:
                debits_q.popleft()
                continue
                
            payment_cents = min(rem_credit_cents, d['remaining_cents'])
            d['remaining_cents'] -= payment_cents
            rem_credit_cents -= payment_cents
            
            event = {
                'date': d['date'],
                'reference': d['reference'],
                'currencyid': d['currencyid'],
                'invoice_amount': d['invoice_amount'],
                'Payment': round(payment_cents / 100.0, 2),
                'remaining': round(d['remaining_cents'] / 100.0, 2),
                'paid_date': credit['date'],
                'aging_days': (credit['date'] - d['date']).days,
                'credit_reference': credit['reference'],
                'is_priority': d['is_priority']
            }
            detailed.append(event)
            
            if d['remaining_cents'] <= 0:
                debits_q.popleft()
    
    # Record unpaid debits
    while debits_q:
        d = debits_q.popleft()
        event = {
            'date': d['date'],
            'reference': d['reference'],
            'currencyid': d['currencyid'],
            'invoice_amount': d['invoice_amount'],
            'Payment': 0.00,
            'remaining': round(d['remaining_cents'] / 100.0, 2),
            'paid_date': None,
            'aging_days': (today - d['date']).days,
            'credit_reference': '-',
            'is_priority': d['is_priority']
        }
        detailed.append(event)
        
    return detailed

# ----------------- Override Functions -----------------
def show_override_selector(raw, start_dt, key="overrides"):
    if raw is None or raw.empty:
        return []

    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    mask = (
        (raw['plantid'] == 56) &
        (raw['date'] > start_dt)
    )
    subset = raw.loc[mask]

    labels = [
        f"{row['functionid']}|{row['recordid']}|{row['date'].date()}|{row['amount']}|{row['reference']}|{row['description']}"
        for _, row in subset.iterrows()
    ]

    return st.multiselect(
        "",
        labels,
        format_func=lambda x: f"Reference: {x.split('|')[4]} - Date: {x.split('|')[2]} - Amount: {x.split('|')[3]}- Description:{x.split('|')[5]}",
        key=key
    )

def apply_overrides(raw, start_dt, chosen):
    for label in chosen:
        try:
            parts = label.split('|')
            if len(parts) != 6:
                continue

            fid = int(parts[0])
            rid = int(parts[1])

            raw.loc[
                (raw['functionid'] == fid) &
                (raw['recordid'] == rid),
                'date'
            ] = start_dt
        except Exception as e:
            st.error(f"خطأ في معالجة العملية: {label} - {str(e)}")
            continue

    return raw

# ----------------- PDF Export Function -----------------
def reshape_text(text):
    if not isinstance(text, str):
        text = str(text)
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception as e:
        print(f"Text reshaping error: {e}")
        return text

class CustomPDF(FPDF):
    def __init__(self, username, execution_datetime, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.username = username
        self.execution_datetime = execution_datetime
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=12)
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)

    def header(self):
        if self.page_no() == 1:
            self.set_font('DejaVu', '', 12)
            title = reshape_text("تفصيلي فترة سداد فواتير عميل")
            self.cell(0, 15, title, ln=1, align='C')
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        username_part = reshape_text(f"User: {self.username}")
        datetime_part = f"Generated on: {self.execution_datetime} | {self.page_no()}/{{nb}}"
        self.cell(0, 10, username_part, 0, 0, 'L')
        self.cell(0, 10, datetime_part, 0, 0, 'R')

def export_pdf(report_df, params):
    execution_time = datetime.now() + timedelta(hours=3)
    execution_datetime = execution_time.strftime('%d/%m/%Y %H:%M:%S')
    username = st.session_state.get('username', 'Unknown User')

    pdf = CustomPDF(username, execution_datetime, orientation='L')
    pdf.add_page()

    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    customer = params.get("اسم العميل")
    if customer is not None:
        line = f"{reshape_text('اسم العميل')}: {customer}"
        pdf.cell(0, 10, line, border=0, ln=1, align='L')
        del params["اسم العميل"]

    params_list = list(params.items())
    half = len(params_list) // 2
    left_params, right_params = params_list[:half], params_list[half:]
    col_width = pdf.w / 2 - 20
    for i in range(max(len(left_params), len(right_params))):
        if i < len(left_params):
            k, v = left_params[i]
            pdf.cell(col_width, 8, f"{reshape_text(k)}: {v}", border=0, align='L')
        if i < len(right_params):
            k, v = right_params[i]
            pdf.cell(col_width, 8, f"{reshape_text(k)}: {v}", border=0, ln=1, align='L')
        else:
            pdf.ln()

    col_widths = [
        30, 40, 30, 35, 30, 35, 32, 30
    ]

    headers = [
        "التاريخ", "الرقم المرجعي",
        "ذهب عيار 21", "تاريخ سداد الذهب",
        "المبلغ النقدي", "تاريخ سداد النقدية",
        "أيام سداد الذهب", "أيام سداد النقدية"
    ]

    def draw_table_headers():
        pdf.set_fill_color(200, 220, 255)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 10, reshape_text(h), border=1, fill=True, align='C')
        pdf.ln()

    draw_table_headers()

    threshold = params.get("فترة سداد العميل", 0)
    row_h = 7
    for _, row in report_df.iterrows():
        if pdf.get_y() + row_h > pdf.h - 15:
            pdf.add_page()
            draw_table_headers()

        cash_age = int(row['aging_days_cash']) if row['aging_days_cash'] not in ('-', '') else 0
        gold_age = int(row['aging_days_gold']) if row['aging_days_gold'] not in ('-', '') else 0

        pdf.set_fill_color(255, 204, 203)

        cells = [
            str(row['date']),
            str(row['reference']),
            f"{float(str(row['amount_gold']).replace(',', '')):,.2f}",
            str(row['paid_date_gold']),
            f"{float(str(row['amount_cash']).replace(',', '')):,.2f}",
            str(row['paid_date_cash']),
            str(row['aging_days_gold']),
            str(row['aging_days_cash']),
        ]

        fills = [
            False, False, False, False, False, False,
            gold_age > threshold, cash_age > threshold
        ]

        for w, text, do_fill in zip(col_widths, cells, fills):
            pdf.cell(w, row_h, reshape_text(text), border=1, fill=do_fill, align='C')
        pdf.ln()
    
    pdf.ln(8)
    pdf.set_font('DejaVu', '', 18)
    pdf.cell(0, 8, "Generated by BI", ln=1, align='R')

    pdf_output = pdf.output(dest='S')
    return bytes(pdf_output) if isinstance(pdf_output, bytearray) else pdf_output

# ----------------- Main Application -----------------
def main():
    st.title("📊 Discount & Payment-Period ByCoustomer Report")
    aging_threshold = st.sidebar.number_input("Enter Aging Days Threshold", min_value=0, value=30, step=1)

    groups = fetch_data("SELECT recordid, name FROM figrp ORDER BY name")
    if groups is None or groups.empty:
        st.error("❌ No groups found or an error occurred while fetching groups.")
        return

    customers = fetch_data("SELECT recordid, name, reference FROM fiacc WHERE groupid = 1")
    cust_list = ["Select Customer..."] + [f"{r['name']} ({r['reference']})" for _, r in customers.iterrows()]
    selected_customer = st.sidebar.selectbox("Customer Name", cust_list)
    start_date = st.sidebar.date_input("Start Date", datetime.now().replace(day=1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    st.sidebar.header("Category Discounts")
    discount_50 = st.sidebar.number_input("احجار عيار 21", 0.0, 1000.0, 0.0)
    discount_61 = st.sidebar.number_input("سادة عيار 21", 0.0, 1000.0, 0.0)
    discount_47 = st.sidebar.number_input("ذهب مشغول عيار 18", 0.0, 1000.0, 0.0)
    discount_62 = st.sidebar.number_input("سادة عيار 18", 0.0, 1000.0, 0.0)
    discount_48 = st.sidebar.number_input("ستار 18", 0.0, 1000.0, 0.0)
    discount_45 = st.sidebar.number_input("تعجيل دفع عيار 21", 0.0, 1000.0, 0.0)
    discount_46 = st.sidebar.number_input("تعجيل دفع عيار 18", 0.0, 1000.0, 0.0)

    raw = None
    if selected_customer != "Select Customer...":
        cid = int(customers.iloc[cust_list.index(selected_customer) - 1]['recordid'])
        query = """
            SELECT f.plantid, f.functionid, f.recordid, f.date, f.reference,f.description,
                   f.currencyid, f.amount, s.qty, s.baseAmount, s.basevatamount, ivit.categoryid
            FROM fitrx f
            LEFT JOIN satrx s ON f.functionid=s.functionid AND f.recordid=s.recordid
            LEFT JOIN ivit ON s.itemid=ivit.recordid
            WHERE f.accountid = :acc
        """
        raw = fetch_data(query, {"acc": cid})
        if raw is None or raw.empty:
            st.warning("لا توجد معاملات متاحة للعميل المحدد.")
            raw = None

    st.markdown("### اختر العمليات خزينة الخصومات:")

    labels = []
    if raw is not None and not raw.empty:
        tmp = raw.copy()
        tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
        mask = (
                (tmp['plantid'] == 56) &
                (tmp['date'] > pd.to_datetime(start_date))
        )
        subset = tmp.loc[mask]
        labels = [
            f"{r['functionid']}|{r['recordid']}|{r['date'].date()}|{r['amount']}|{r['reference']} |{r['description']}"
            for _, r in subset.iterrows()
        ]

    overrides = st.multiselect(
        "",
        options=labels,
        format_func=lambda x: f"Reference: {x.split('|')[4]} – Date: {x.split('|')[2]} – Amount: {x.split('|')[3]} - Description:{x.split('|')[5]}",
        key="overrides_pre_generate"
    )

    if len(labels) > 0 and len(overrides) != len(labels):
        st.error(f"⛔ يجب اختيار جميع العمليات ({len(labels)}) قبل عمل Generate.")
        return

    if st.sidebar.button("Generate Report"):
        if selected_customer == "Select Customer...":
            st.error("الرجاء اختيار عميل.")
            return
        if raw is None or raw.empty:
            st.warning("لا توجد معاملات للعميل المحدد.")
            return

        start_time = time.time()
        discounts = {50: discount_50, 47: discount_47, 61: discount_61, 62: discount_62, 48: discount_48}
        extras = {
            50: discount_45,
            61: discount_45,
            47: discount_46,
            62: discount_46
        }
        raw2 = raw.copy()
        raw2['date'] = pd.to_datetime(raw2['date'], errors='coerce')

        # Apply overrides without removing plantid=56 transactions
        for label in overrides:
            fid, rid, *_ = label.split('|')
            raw2.loc[
                (raw2['functionid'] == int(fid)) & (raw2['recordid'] == int(rid)),
                'date'
            ] = pd.to_datetime(start_date)

        txs = process_transactions(raw2, discounts, extras, pd.to_datetime(start_date))
        if txs.empty:
            st.warning("لا توجد معاملات بعد المعالجة.")
            return

        # Generate Aggregated Aging Report (excludes priority transactions)
        report = calculate_aging_reports(txs)
        report = report[pd.to_datetime(report['date']) >= pd.to_datetime("2023-01-01")]
        report['date_dt'] = pd.to_datetime(report['date'])
        report = report[(report['date_dt'] >= pd.to_datetime(start_date)) & (report['date_dt'] <= pd.to_datetime(end_date))]
        report = report.sort_values(by=['date_dt', 'paid_date_cash', 'paid_date_gold'],
                                    ascending=[True, True, True]).reset_index(drop=True)
        report = report.drop(columns=['date_dt'])
        for col in ['amount_cash', 'remaining_cash', 'amount_gold', 'remaining_gold']:
            report[col] = report[col].apply(lambda x: f"{x:,.2f}")

        def highlight_row(row):
            styles = [''] * len(row)
            try:
                cash = int(row['aging_days_cash']) if row['aging_days_cash'] != '-' else 0
                gold = int(row['aging_days_gold']) if row['aging_days_gold'] != '-' else 0
            except:
                cash = gold = 0
            if cash > aging_threshold and gold > aging_threshold:
                styles = ['background-color: #FFCCCB'] * len(row)
            else:
                if cash > aging_threshold:
                    idx = row.index.get_loc('aging_days_cash')
                    styles[idx] = 'background-color: #FFCCCB'
                if gold > aging_threshold:
                    idx = row.index.get_loc('aging_days_gold')
                    styles[idx] = 'background-color: #FFCCCB'
            return styles

        styled_report = report.style.apply(highlight_row, axis=1)
        st.subheader("Aging Report")
        st.dataframe(styled_report, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col2:
            report_params = {
                "اسم العميل": reshape_text(selected_customer),
                "تاريخ البداية": str(start_date),
                "تاريخ النهاية": str(end_date),
                "فترة سداد العميل": aging_threshold,
                "احجار عيار 21": discount_50,
                "سادة عيار 21": discount_61,
                "ذهب مشغول عيار 18": discount_47,
                "سادة عيار 18": discount_62,
                "ستار 18": discount_48,
                "تعجيل دفع عيار 21": discount_45,
                "تعجيل دفع عيار 18": discount_46
            }
            pdf_bytes = export_pdf(report, report_params)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download Report",
                    data=pdf_bytes,
                    file_name="تقرير_الخصومات.pdf",
                    mime="application/pdf"
                )

        # Detailed Installments Search by Reference
        st.markdown("---")
        st.subheader("تفاصيل سداد فاتورة معينة")

        # Build detailed FIFO events for installments (includes priority transactions)
        cash_debits, cash_credits, gold_debits, gold_credits = [], [], [], []
        fioba = fetch_data(
            "SELECT fiscalYear, currencyid, amount FROM fioba WHERE fiscalYear = 2023 AND accountId = :acc",
            {"acc": cid}
        )
        if fioba is not None and not fioba.empty:
            for _, r in fioba.iterrows():
                entry_date = pd.to_datetime(f"{int(r['fiscalYear'])}-01-01")
                conv = r['amount']
                if r['currencyid'] != 1:
                    conv = convert_gold({'reference': '', 'amount': r['amount'], 'currencyid': r['currencyid']})
                entry = {
                    'date': entry_date,
                    'reference': 'Opening-Balance-2023',
                    'currencyid': r['currencyid'],
                    'amount': abs(conv),
                    'remaining': abs(conv),
                    'is_priority': False
                }
                if entry_date >= pd.to_datetime("2023-01-01"):
                    if conv >= 0:
                        if r['currencyid'] == 1:
                            cash_debits.append(entry)
                        else:
                            gold_debits.append(entry)
                    else:
                        if r['currencyid'] == 1:
                            cash_credits.append({'date': entry_date, 'amount': abs(conv)})
                        else:
                            gold_credits.append({'date': entry_date, 'amount': abs(conv)})
        for _, r in txs.iterrows():
            if r['date'] < pd.to_datetime("2023-01-01"):
                continue
            entry = {
                'date': r['date'],
                'reference': r['reference'],
                'currencyid': r['currencyid'],
                'amount': abs(r['converted']),
                'remaining': abs(r['converted']),
                'is_priority': r['is_priority']
            }
            if r['amount'] > 0:
                if r['currencyid'] == 1:
                    cash_debits.append(entry)
                else:
                    gold_debits.append(entry)
            else:
                if r['currencyid'] == 1:
                    cash_credits.append({
                        'date': r['date'], 
                        'amount': abs(r['converted']),
                        'reference': r['reference']
                    })
                else:
                    gold_credits.append({
                        'date': r['date'], 
                        'amount': abs(r['converted']),
                        'reference': r['reference']
                    })
                    
        cash_details = process_fifo_detailed(cash_debits, cash_credits)
        gold_details = process_fifo_detailed(gold_debits, gold_credits)
        cash_details_df = pd.DataFrame(cash_details)
        gold_details_df = pd.DataFrame(gold_details)

        if not cash_details_df.empty:
            cash_details_df['date'] = pd.to_datetime(cash_details_df['date'])
            cash_details_df = cash_details_df[(cash_details_df['date'] >= pd.to_datetime(start_date)) &
                                              (cash_details_df['date'] <= pd.to_datetime(end_date))]
        if not gold_details_df.empty:
            gold_details_df['date'] = pd.to_datetime(gold_details_df['date'])
            gold_details_df = gold_details_df[(gold_details_df['date'] >= pd.to_datetime(start_date)) &
                                              (gold_details_df['date'] <= pd.to_datetime(end_date))]
        # Format amounts for display
        if not cash_details_df.empty:
            cash_details_df['Remaining %'] = cash_details_df.apply(
                lambda r: (r['remaining'] / r['invoice_amount'] * 100) if r['invoice_amount'] != 0 else 0, axis=1
            )
            cash_details_df['invoice_amount'] = cash_details_df['invoice_amount'].apply(lambda x: f"{x:,.2f}")
            cash_details_df['Payment'] = cash_details_df['Payment'].apply(lambda x: f"{x:,.2f}")
            cash_details_df['remaining'] = cash_details_df['remaining'].apply(lambda x: f"{x:,.2f}")
            cash_details_df['Remaining %'] = cash_details_df['Remaining %'].apply(lambda x: f"{x:,.2f}")
        if not gold_details_df.empty:
            gold_details_df['Remaining %'] = gold_details_df.apply(
                lambda r: (r['remaining'] / r['invoice_amount'] * 100) if r['invoice_amount'] != 0 else 0, axis=1
            )
            gold_details_df['invoice_amount'] = gold_details_df['invoice_amount'].apply(lambda x: f"{x:,.2f}")
            gold_details_df['Payment'] = gold_details_df['Payment'].apply(lambda x: f"{x:,.2f}")
            gold_details_df['remaining'] = gold_details_df['remaining'].apply(lambda x: f"{x:,.2f}")
            gold_details_df['Remaining %'] = gold_details_df['Remaining %'].apply(lambda x: f"{x:,.2f}")
        if not cash_details_df.empty:
            cash_details_df['Invoice Date'] = cash_details_df['date'].dt.strftime('%Y-%m-%d')
            cash_details_df['Paid Date'] = cash_details_df['paid_date'].apply(
                lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else "Unpaid")
        if not gold_details_df.empty:
            gold_details_df['Invoice Date'] = gold_details_df['date'].dt.strftime('%Y-%m-%d')
            gold_details_df['Paid Date'] = gold_details_df['paid_date'].apply(
                lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else "Unpaid")
            
        st.markdown("### تفاصيل سداد الذهب")
        if not gold_details_df.empty:
            st.dataframe(gold_details_df[
                             ['Invoice Date', 'reference', 'invoice_amount', 'Payment', 'remaining', 
                              'Remaining %', 'Paid Date', 'aging_days', 'is_priority', 'credit_reference']
                         ].reset_index(drop=True), use_container_width=True)
        else:
            st.info("لا توجد بيانات سداد ذهباً لهذه الفاتورة.")
        st.markdown("### تفاصيل سداد النقدية")
        if not cash_details_df.empty:
            st.dataframe(cash_details_df[
                             ['Invoice Date', 'reference', 'invoice_amount', 'Payment', 'remaining', 
                              'Remaining %', 'Paid Date', 'aging_days', 'is_priority', 'credit_reference']
                         ].reset_index(drop=True), use_container_width=True)
        else:
            st.info("لا توجد بيانات سداد نقداً لهذه الفاتورة.")

if __name__ == "__main__":
    main()
