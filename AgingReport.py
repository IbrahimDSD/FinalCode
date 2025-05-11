import pandas as pd
import numpy as np
from datetime import date, datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from collections import deque
from passlib.hash import pbkdf2_sha256
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import BytesIO
import sqlitecloud
import logging
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQLite Cloud database connection details
USER_DB_URI = (
    "sqlitecloud://cpran7d0hz.g2.sqlite.cloud:8860/"
    "user_management.db?apikey=oUEez4Dc0TFsVVIVFu8SDRiXea9YVQLOcbzWBsUwZ78"
)

# ----------------- Authentication Setup -----------------
def get_connection():
    logger.debug("Attempting SQLite Cloud connection")
    try:
        conn = sqlitecloud.connect(USER_DB_URI)
        logger.debug("Connection successful")
        return conn
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return None

def get_user_record(username: str):
    conn = get_connection()
    if conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, password_hash, permissions, role FROM users WHERE username = ?",
            (username,)
        )
        rec = c.fetchone()
        conn.close()
        logger.debug(f"User record for {username}: {rec}")
        return rec
    logger.warning("No connection for get_user_record")
    return None

def check_login(username: str, password: str) -> bool:
    rec = get_user_record(username)
    if not rec:
        logger.warning(f"No user found for {username}")
        return False
    user_id, pw_hash, permissions, role = rec
    result = pbkdf2_sha256.verify(password, pw_hash)
    logger.debug(f"Login check for {username}: {result}")
    return result

# ----------------- Helper Functions -----------------
def reshape_text(txt):
    return get_display(arabic_reshaper.reshape(str(txt)))

def create_db_engine():
    server = "52.48.117.197"
    database = "R1029"
    username = "sa"
    password = "Argus@NEG"
    driver = "ODBC Driver 17 for SQL Server"
    odbc = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        f"UID={username};PWD={password};TrustServerCertificate=Yes;"
    )
    url = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"
    logger.debug("Attempting SQL Server connection")
    try:
        eng = create_engine(url, connect_args={"timeout": 5})
        with eng.connect():
            logger.debug("SQL Server connection successful")
        return eng, None
    except Exception as e:
        logger.error(f"SQL Server connection error: {e}")
        return None, str(e)

def convert_gold(cur, amt):
    if cur == 2:   return amt * 6.0 / 7.0
    if cur == 3:   return amt
    if cur == 4:   return amt * 24.0 / 21.0
    if cur == 14:  return amt * 14.0 / 21.0
    return amt

PRIORITY_FIDS = {3001, 3100, 3108, 3113, 3104}
def process_fifo(debits, credits, as_of, priority_fids=PRIORITY_FIDS):
    credits = [c for c in credits if c["date"] <= as_of]
    pri = deque(sorted(
        [d for d in debits if d["date"] <= as_of and d["functionid"] in priority_fids],
        key=lambda x: (x["date"], x["invoiceref"])
    ))
    reg = deque(sorted(
        [d for d in debits if d["date"] <= as_of and d["functionid"] not in priority_fids],
        key=lambda x: (x["date"], x["invoiceref"])
    ))

    excess = 0.0
    for cr in sorted(credits, key=lambda x: (x["date"], x.get("invoiceref", ""))):
        rem = cr["amount"]
        while rem > 0 and pri:
            d = pri[0]
            ap = min(rem, d["remaining"])
            d["remaining"] -= ap
            rem -= ap
            if d["remaining"] <= 0:
                d["paid_date"] = cr["date"]
                pri.popleft()
        while rem > 0 and not pri and reg:
            d = reg[0]
            ap = min(rem, d["remaining"])
            d["remaining"] -= ap
            rem -= ap
            if d["remaining"] <= 0:
                d["paid_date"] = cr["date"]
                reg.popleft()
        excess += rem

    remaining = list(pri) + list(reg)
    total_remaining = sum(d["remaining"] for d in remaining)
    net_balance = total_remaining - excess
    logger.debug(f"Process FIFO: {len(remaining)} remaining debits, net balance: {net_balance}")
    return remaining, net_balance

def bucketize(days, grace, length):
    if days <= grace:
        return None
    adj = days - grace
    if adj <= length:
        return f"{grace + 1}-{grace + length}"
    if adj <= 2 * length:
        return f"{grace + length + 1}-{grace + 2 * length}"
    if adj <= 3 * length:
        return f"{grace + 2 * length + 1}-{grace + 3 * length}"
    return f">{grace + 3 * length}"

def format_number(value):
    try:
        value = round(float(value), 3)
        if value < 0:
            return f"({abs(value):,.3f})"
        elif value == 0:
            return "-"
        else:
            return f"{value:,.3f}"
    except (ValueError, TypeError):
        return str(value)

# ----------------- Data Fetching Functions -----------------
def get_salespersons(engine):
    df = pd.read_sql("SELECT recordid, name FROM sasp ORDER BY name", engine)
    logger.debug(f"Salespersons fetched: {len(df)} rows")
    return df

def get_customers(engine, sp_id):
    if sp_id is None:
        sql = """
            SELECT DISTINCT acc.recordid, acc.name, acc.spid, COALESCE(sasp.name, '') AS sp_name
            FROM fiacc acc
            LEFT JOIN sasp ON acc.spid = sasp.recordid
            WHERE acc.groupid = 1
            ORDER BY acc.name
            """
        df = pd.read_sql(text(sql), engine)
    else:
        sql = """
            SELECT DISTINCT acc.recordid, acc.name, acc.spid, sasp.name AS sp_name
            FROM fiacc acc
            JOIN sasp ON acc.spid = sasp.recordid
            WHERE acc.spid = :sp
            ORDER BY acc.name
            """
        df = pd.read_sql(text(sql), engine, params={"sp": sp_id})
    logger.debug(f"Customers fetched: {len(df)} rows")
    return df

def get_overdues(engine, sp_id, as_of, grace, length):
    base_sql = """
        SELECT f.accountid,
               f.functionid,
               acc.reference AS code,
               acc.name AS name,
               f.currencyid,
               f.amount,
               f.date,
               COALESCE(f.reference, CAST(f.date AS VARCHAR)) AS invoiceref,
               acc.spid,
               COALESCE(sasp.name,'غير محدد') AS sp_name
        FROM fitrx f
        JOIN fiacc acc ON f.accountid = acc.recordid
        LEFT JOIN sasp ON acc.spid = sasp.recordid
        WHERE acc.groupid = 1 AND f.date <= :as_of
    """
    params = {"as_of": as_of}
    if sp_id is not None:
        base_sql += " AND acc.spid = :sp"
        params["sp"] = sp_id
    base_sql += " ORDER BY acc.reference, f.date"

    logger.debug(f"Executing get_overdues with sp_id: {sp_id}, as_of: {as_of}")
    raw = pd.read_sql(text(base_sql), engine, params=params)
    logger.debug(f"Raw data fetched: {len(raw)} rows")

    buckets = [
        f"{grace + 1}-{grace + length}",
        f"{grace + length + 1}-{grace + 2 * length}",
        f"{grace + 2 * length + 1}-{grace + 3 * length}",
        f">{grace + 3 * length}"
    ]
    summary_rows = []
    invoice_data = []

    for acc, grp in raw.groupby("accountid"):
        code = grp["code"].iat[0]
        name = grp["name"].iat[0]
        spid = grp["spid"].iat[0]
        sp_name = grp["sp_name"].iat[0]

        cash_debits = []
        cash_credits = []
        gold_debits = []
        gold_credits = []

        for _, r in grp.iterrows():
            dt = pd.to_datetime(r["date"])
            amt = r["amount"]
            invoiceref = r["invoiceref"]
            fid = r["functionid"]

            if r["currencyid"] == 1:
                if amt > 0:
                    cash_debits.append({
                        "date": dt,
                        "remaining": amt,
                        "paid_date": None,
                        "original_amount": amt,
                        "invoiceref": invoiceref,
                        "functionid": fid
                    })
                else:
                    cash_credits.append({"date": dt, "amount": abs(amt)})
            else:
                grams = convert_gold(r["currencyid"], amt)
                if amt > 0:
                    gold_debits.append({
                        "date": dt,
                        "remaining": grams,
                        "paid_date": None,
                        "original_amount": grams,
                        "invoiceref": invoiceref,
                        "functionid": fid
                    })
                else:
                    gold_credits.append({"date": dt, "amount": abs(grams)})

        pc, net_cash = process_fifo(cash_debits, cash_credits, pd.to_datetime(as_of))
        pg, net_gold = process_fifo(gold_debits, gold_credits, pd.to_datetime(as_of))

        sums = {f"cash_{b}": 0.0 for b in buckets}
        sums.update({f"gold_{b}": 0.0 for b in buckets})
        inv_over = {}

        for drv, net, pfx in [(pc, net_cash, "cash"), (pg, net_gold, "gold")]:
            for d in drv:
                if d["remaining"] > 0:
                    days = ((d.get("paid_date") or pd.to_datetime(as_of)) - d["date"]).days
                    bucket = bucketize(days, grace, length)
                    if bucket:
                        sums[f"{pfx}_{bucket}"] += d["remaining"]
                        ref = d["invoiceref"]
                        if ref not in inv_over:
                            inv_over[ref] = {
                                "Customer Reference": code,
                                "Customer Name": name,
                                "Invoice Ref": ref,
                                "Invoice Date": d["date"].date(),
                                "Overdue G21": 0.0,
                                "Overdue EGP": 0.0,
                                "Delay Days": max(0, days - grace)
                            }
                        inv_over[ref][f"Overdue {'G21' if pfx=='gold' else 'EGP'}"] += d["remaining"]

        invoice_data.extend(inv_over.values())

        cash_total = sum(sums[f"cash_{b}"] for b in buckets)
        gold_total = sum(sums[f"gold_{b}"] for b in buckets)
        if cash_total > 0 or gold_total > 0:
            summary_rows.append({
                "AccountID": acc,
                "Customer": name,
                "Code": code,
                "sp_name": sp_name,
                "spid": spid,
                "total_cash_due": net_cash,
                "total_gold_due": net_gold,
                **sums,
                "cash_total": cash_total,
                "gold_total": gold_total
            })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(invoice_data)
    if not detail_df.empty:
        detail_df.sort_values(
            by=["Invoice Date", "Invoice Ref"],
            key=lambda col: col.astype(str),
            inplace=True
        )
    logger.debug(f"Summary DF: {len(summary_df)} rows, Detail DF: {len(detail_df)} rows")
    return summary_df, buckets, detail_df

# ----------------- Chart Generation Functions -----------------
def setup_arabic_font():
    font_path = "DejaVuSans.ttf"
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family='DejaVu Sans')
        logger.debug("Arabic font loaded successfully")
        return prop
    logger.warning("Arabic font not found")
    return None

def create_pie_chart(summary_df, buckets, type="cash"):
    total_overdues = {b: summary_df[f"{type}_{b}"].sum() for b in buckets}
    total = sum(total_overdues.values())
    logger.debug(f"Pie chart {type} total: {total}")
    if total == 0:
        logger.warning(f"No {type} overdues to plot")
        return None

    sizes = [total_overdues[b] for b in buckets]
    outer_labels = [reshape_text(f"الفترة: {b}") for b in buckets]
    prop = setup_arabic_font()
    plt.figure(figsize=(8, 4))
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=outer_labels,
        startangle=140,
        labeldistance=1.1,
        pctdistance=0.65,
        autopct=lambda pct: reshape_text(f"{format_number(pct * total / 100)}\n{pct:.1f}%"),
        textprops={'fontproperties': prop, 'fontsize': 8} if prop else {'fontsize': 8},
    )
    for txt in autotexts:
        txt.set_color('white')
        if prop:
            txt.set_fontproperties(prop)
        txt.set_fontsize(9)
    title = "توزيع التأخيرات حسب الفترة " + ("(كاش)" if type == "cash" else "(ذهب)")
    plt.title(reshape_text(title), fontproperties=prop, fontsize=11 if prop else None)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    logger.debug(f"Pie chart {type} generated")
    return buf

def create_bar_chart(summary_df, buckets, type="cash"):
    df = summary_df.copy()
    df["total_overdue"] = df[f"{type}_total"]
    top_10 = df.nlargest(10, "total_overdue")
    total = top_10["total_overdue"].sum()
    logger.debug(f"Bar chart {type} total: {total}")
    if total == 0:
        logger.warning(f"No {type} overdues to plot")
        return None

    customers = top_10["Customer"]
    overdues = top_10["total_overdue"]
    labels = [reshape_text(c) for c in customers]
    prop = setup_arabic_font()
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range
    bars = plt.bar(range(len(labels)), overdues, tick_label=labels)
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.02 * overdues.max(),
            format_number(y),
            ha='center', va='bottom',
            fontproperties=prop, fontsize=9 if prop else None,
        )
    title = f"أعلى 10 عملاء بالمتأخرات ({'كاش' if type == 'cash' else 'ذهب'})"
    plt.title(reshape_text(title), fontproperties=prop, fontsize=11 if prop else None)
    plt.xticks(rotation=45, ha="right", fontproperties=prop, fontsize=9 if prop else None)
    plt.yticks([])
    plt.ylabel("")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35, left=0.1)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    logger.debug(f"Bar chart {type} generated")
    return buf

# ----------------- PDF Generation Functions -----------------
def truncate_text(pdf, text, width):
    ellipsis = "..."
    while pdf.get_string_width(ellipsis + text) > width and len(text) > 0:
        text = text[1:]
    if pdf.get_string_width(ellipsis + text) <= width:
        text = ellipsis + text
    return text

def draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, sub_w):
    pdf.cell(name_w, 8, reshape_text("Name"), border=1, align="C", ln=0)
    pdf.cell(bal_w, 8, reshape_text("Balance"), border=1, align="C", ln=0)
    for b in buckets:
        pdf.cell(bucket_w, 8, reshape_text(f"From {b.replace('-', ' - ')}"), border=1, align="C", ln=0)
    pdf.cell(tot_w, 8, reshape_text("Total Delay"), border=1, align="C", ln=1)
    pdf.cell(name_w, 8, "", border=1, ln=0)
    pdf.cell(sub_w, 8, "G21", border=1, align="C", ln=0)
    pdf.cell(sub_w, 8, "EGP", border=1, align="C", ln=0)
    for _ in buckets:
        pdf.cell(sub_w, 8, "G21", border=1, align="C", ln=0)
        pdf.cell(sub_w, 8, "EGP", border=1, align="C", ln=0)
    pdf.cell(sub_w, 8, "G21", border=1, align="C", ln=0)
    pdf.cell(sub_w, 8, "EGP", border=1, align="C", ln=1)

def draw_parameters_table(pdf, sp_name, selected_customer, as_of, grace, length, table_width, col_widths):
    parameters = [
        ("Sales Person", sp_name),
        ("Customer Name", selected_customer),
        ("Due Date", as_of.strftime('%d/%m/%Y')),
        ("Grace Period", f"{grace} يوم"),
        ("Period Length", f"{length} يوم")
    ]
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(col_widths[0], 8, reshape_text("المعامل"), border=1, align="C", fill=True, ln=0)
    pdf.cell(col_widths[1], 8, reshape_text("القيمة"), border=1, align="C", fill=True, ln=1)
    for label, value in parameters:
        pdf.cell(col_widths[0], 8, reshape_text(label), border=1, align="R", ln=0)
        pdf.cell(col_widths[1], 8, reshape_text(value), border=1, align="R", ln=1)

def build_summary_pdf(df, sp_name, as_of, buckets, selected_customer, grace, length):
    pdf = FPDF(orientation="L", unit="mm", format="A3")
    pdf.add_page()
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        logger.error("Font file DejaVuSans.ttf not found")
        return None
    pdf.add_font('DejaVu', '', font_path, uni=True)
    pdf.set_font('DejaVu', '', 12)

    exe = datetime.now().strftime("%d/%m/%Y %I:%M %p")
    pdf.set_xy(10, 10)
    pdf.cell(0, 5, reshape_text("New Egypt Gold | تقرير المتأخرات"), ln=0, align="C")
    pdf.ln(5)
    pdf.cell(0, 5, f"Execution Date: {exe}", ln=0, align="L")
    pdf.ln(10)

    table_width = 120
    col_widths = [40, 80]
    pdf.set_xy(10, pdf.get_y())
    draw_parameters_table(pdf, sp_name, selected_customer, as_of, grace, length, table_width, col_widths)
    pdf.ln(10)

    name_w = 50
    bal_w = 60
    bucket_w = 60
    tot_w = 60
    sub_w = bal_w / 2
    line_h = 7
    bottom_margin = 20

    grouped = [(sp_name, df)] if sp_name != "All" else df.groupby("sp_name", dropna=False)

    for sp_id, group in grouped:
        try:
            sp_display_name = group["sp_name"].iloc[0] if sp_name == "All" else sp_name
            if sp_id in (0, '0', None):
                sp_display_name = "غير محدد"

            pdf.set_xy(10, pdf.get_y())
            pdf.cell(0, 5, reshape_text(f"Sales Person: {sp_display_name}"), border=0, ln=1, align="L")
            pdf.ln(4)
            draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, sub_w)

            for _, r in group.iterrows():
                row_h = line_h
                g21_h = pdf.multi_cell(sub_w, line_h, format_number(r["total_gold_due"]), border=0, align="R", dry_run=True)
                row_h = max(row_h, g21_h)
                egp_h = pdf.multi_cell(sub_w, line_h, format_number(r["total_cash_due"]), border=0, align="R", dry_run=True)
                row_h = max(row_h, egp_h)
                for b in buckets:
                    gold_h = pdf.multi_cell(sub_w, line_h, format_number(r[f"gold_{b}"]), border=0, align="R", dry_run=True)
                    cash_h = pdf.multi_cell(sub_w, line_h, format_number(r[f"cash_{b}"]), border=0, align="R", dry_run=True)
                    row_h = max(row_h, gold_h, cash_h)
                tot_g21_h = pdf.multi_cell(sub_w, line_h, format_number(r["gold_total"]), border=0, align="R", dry_run=True)
                tot_egp_h = pdf.multi_cell(sub_w, line_h, format_number(r["cash_total"]), border=0, align="R", dry_run=True)
                row_h = max(row_h, tot_g21_h, tot_egp_h)

                if pdf.get_y() + row_h + bottom_margin > pdf.h:
                    pdf.add_page()
                    pdf.add_font('DejaVu', '', font_path, uni=True)
                    pdf.set_font('DejaVu', '', 12)
                    pdf.cell(0, 5, reshape_text(f"Sales Person: {sp_display_name}"), border=0, ln=1, align="L")
                    pdf.ln(4)
                    draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, sub_w)

                x0, y0 = pdf.get_x(), pdf.get_y()
                heights = []
                customer_name = reshape_text(r["Customer"])
                if pdf.get_string_width(customer_name) > name_w - 2:
                    customer_name = truncate_text(pdf, customer_name, name_w - 2)
                pdf.cell(name_w, line_h, customer_name, border=1, align="L", ln=0)
                heights.append(line_h)

                pdf.set_xy(x0 + name_w, y0)
                color = (0, 128, 0) if r["total_gold_due"] <= 0 else (0, 0, 255)
                pdf.set_text_color(*color)
                pdf.multi_cell(sub_w, line_h, format_number(r["total_gold_due"]), border=1, align="R")
                heights.append(pdf.get_y() - y0)

                pdf.set_xy(x0 + name_w + sub_w, y0)
                color = (0, 128, 0) if r["total_cash_due"] <= 0 else (255, 0, 0)
                pdf.set_text_color(*color)
                pdf.multi_cell(sub_w, line_h, format_number(r["total_cash_due"]), border=1, align="R")
                heights.append(pdf.get_y() - y0)
                pdf.set_text_color(0, 0, 0)

                x_b = x0 + name_w + bal_w
                for i, b in enumerate(buckets):
                    pdf.set_xy(x_b + i * bucket_w, y0)
                    pdf.multi_cell(sub_w, line_h, format_number(r[f"gold_{b}"]), border=1, align="R")
                    heights.append(pdf.get_y() - y0)
                    pdf.set_xy(x_b + i * bucket_w + sub_w, y0)
                    pdf.multi_cell(sub_w, line_h, format_number(r[f"cash_{b}"]), border=1, align="R")
                    heights.append(pdf.get_y() - y0)

                x_t = x_b + len(buckets) * bucket_w
                pdf.set_xy(x_t, y0)
                pdf.multi_cell(sub_w, line_h, format_number(r["gold_total"]), border=1, align="R")
                heights.append(pdf.get_y() - y0)
                pdf.set_xy(x_t + sub_w, y0)
                pdf.multi_cell(sub_w, line_h, format_number(r["cash_total"]), border=1, align="R")
                heights.append(pdf.get_y() - y0)

                row_h = max(heights)
                pdf.set_xy(x0, y0 + row_h)

            pdf.ln(10)

        except Exception as e:
            logger.error(f"Error in PDF generation: {str(e)}")
            return None

    out = pdf.output(dest="S")
    logger.debug(f"Summary PDF generated: {len(out) if isinstance(out, bytes) else 'None'} bytes")
    return bytes(out) if isinstance(out, bytearray) else out

def build_detailed_pdf(detail_df, summary_df, sp_name, as_of, selected_customer, grace, length):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        logger.error("Font file DejaVuSans.ttf not found")
        return None
    pdf.add_font('DejaVu', '', font_path, uni=True)
    pdf.set_font('DejaVu', '', 12)

    execution_date = datetime.now().strftime("%d/%m/%Y %H:%M %p")
    pdf.set_xy(10, 10)
    pdf.cell(0, 5, reshape_text(f"New Egypt Gold | تقرير تفصيلي للمتأخرات"), border=0, ln=0, align="R")

    table_width = 120
    col_widths = [40, 80]
    pdf.set_xy(10, pdf.get_y())
    draw_parameters_table(pdf, sp_name, selected_customer, as_of, grace, length, table_width, col_widths)
    pdf.ln(10)

    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 8, reshape_text("Customer Delays By Custom Range."), border=1, ln=1, align="C", fill=True)
    pdf.cell(30, 5, reshape_text("Due Date:"), border=0, ln=0, align="L")
    pdf.cell(30, 5, as_of.strftime("%d/%m/%Y"), border=0, ln=0, align="L")
    pdf.ln(5)

    customers = set(summary_df["Customer"])
    for customer in sorted(customers):
        group = detail_df[detail_df["Customer Name"] == customer]
        if not group.empty:
            customer_summary = summary_df[summary_df["Customer"] == customer]
            total_cash_due = customer_summary["total_cash_due"].iloc[0] if not customer_summary.empty else 0.0
            total_gold_due = customer_summary["total_gold_due"].iloc[0] if not customer_summary.empty else 0.0
            total_cash_overdue = customer_summary["cash_total"].iloc[0] if not customer_summary.empty else 0.0
            total_gold_overdue = customer_summary["gold_total"].iloc[0] if not customer_summary.empty else 0.0

            pdf.set_xy(10, pdf.get_y())
            pdf.multi_cell(0, 5, reshape_text(f"العميل: {customer}"), border=0, align="R")
            pdf.set_xy(10, pdf.get_y())
            pdf.set_text_color(0, 128, 0) if total_cash_due <= 0 else pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 5, reshape_text(f"إجمالي المديونية النقدية: {format_number(total_cash_due)}"), border=0, ln=1, align="R")
            pdf.set_text_color(0, 128, 0) if total_gold_due <= 0 else pdf.set_text_color(0, 0, 255)
            pdf.cell(0, 5, reshape_text(f"إجمالي المديونية الذهبية: {format_number(total_gold_due)}"), border=0, ln=1, align="R")
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 5, reshape_text(f"إجمالي المتأخرات النقدية: {format_number(total_cash_overdue)}"), border=0, ln=1, align="R")
            pdf.cell(0, 5, reshape_text(f"إجمالي المتأخرات الذهبية: {format_number(total_gold_overdue)}"), border=0, ln=1, align="R")
            pdf.ln(4)

            headers = ["رقم الفاتورة", "تاريخ الفاتورة", "المتأخرة G21", "المتأخرة EGP", "عدد أيام التأخير"]
            widths = [40, 40, 30, 30, 30]
            for w, h in zip(widths, headers):
                pdf.cell(w, 8, reshape_text(h), border=1, ln=0, align="C")
            pdf.ln()
            for _, row in group.iterrows():
                pdf.cell(40, 10, reshape_text(row["Invoice Ref"]), border=1, align="C", ln=0)
                pdf.cell(40, 10, str(row["Invoice Date"]), border=1, align="C", ln=0)
                pdf.cell(30, 10, format_number(row["Overdue G21"]), border=1, align="R", ln=0)
                pdf.cell(30, 10, format_number(row["Overdue EGP"]), border=1, align="R", ln=0)
                pdf.cell(30, 10, str(row["Delay Days"]), border=1, align="R", ln=1)
            pdf.ln(4)

    pdf_output = pdf.output(dest='S')
    logger.debug(f"Detailed PDF generated: {len(pdf_output) if isinstance(pdf_output, bytes) else 'None'} bytes")
    return bytes(pdf_output) if isinstance(pdf_output, bytearray) else pdf_output

# ----------------- Test Function -----------------
def test_overdues():
    # Test parameters
    username = "test_user"  # Replace with a valid username
    password = "test_password"  # Replace with a valid password
    sp_id = None  # None for "All" salespersons
    selected_customer = "الكل"  # "الكل" for all customers
    as_of = date.today()
    grace = 30
    length = 15

    # Test login
    logger.info("Testing login")
    if check_login(username, password):
        logger.info("Login successful")
    else:
        logger.error("Login failed")
        return

    # Test database connection
    logger.info("Testing SQL Server connection")
    engine, err = create_db_engine()
    if engine is None:
        logger.error(f"Failed to create engine: {err}")
        return

    # Test salespersons
    logger.info("Fetching salespersons")
    sps = get_salespersons(engine)
    if sps.empty:
        logger.warning("No salespersons found")
        return
    logger.info(f"Salespersons: {sps['name'].tolist()}")

    # Test customers
    logger.info("Fetching customers")
    customers = get_customers(engine, sp_id)
    if customers.empty:
        logger.warning("No customers found")
        return
    logger.info(f"Customers: {len(customers)} found")

    # Test overdues
    logger.info("Fetching overdues")
    summary_df, buckets, detail_df = get_overdues(engine, sp_id, as_of, grace, length)
    if summary_df.empty:
        logger.warning("No overdue data found")
        return
    logger.info(f"Summary DF rows: {len(summary_df)}, Detail DF rows: {len(detail_df)}")
    logger.debug(f"Summary DF columns: {summary_df.columns.tolist()}")

    # Filter by customer if needed
    if selected_customer != "الكل":
        summary_df = summary_df[summary_df["Customer"] == selected_customer]
        detail_df = detail_df[detail_df["Customer Name"] == selected_customer]
        logger.info(f"Filtered for customer {selected_customer}: {len(summary_df)} summary rows, {len(detail_df)} detail rows")

    if summary_df.empty:
        logger.warning("No data after customer filter")
        return

    # Test pie chart
    logger.info("Generating cash pie chart")
    pie_chart_cash = create_pie_chart(summary_df, buckets, type="cash")
    if pie_chart_cash is None:
        logger.warning("Cash pie chart returned None")
    else:
        with open("pie_chart_cash.png", "wb") as f:
            f.write(pie_chart_cash.read())
        logger.info("Cash pie chart saved as pie_chart_cash.png")

    # Test bar chart
    logger.info("Generating cash bar chart")
    bar_chart_cash = create_bar_chart(summary_df, buckets, type="cash")
    if bar_chart_cash is None:
        logger.warning("Cash bar chart returned None")
    else:
        with open("bar_chart_cash.png", "wb") as f:
            f.write(bar_chart_cash.read())
        logger.info("Cash bar chart saved as bar_chart_cash.png")

    # Test summary PDF
    logger.info("Generating summary PDF")
    pdf_summary = build_summary_pdf(summary_df, "All" if sp_id is None else sps.loc[sps["recordid"] == sp_id, "name"].iloc[0], 
                                    as_of, buckets, selected_customer, grace, length)
    if pdf_summary is None:
        logger.error("Summary PDF returned None")
    else:
        with open("summary_overdues.pdf", "wb") as f:
            f.write(pdf_summary)
        logger.info("Summary PDF saved as summary_overdues.pdf")

    # Test detailed PDF
    logger.info("Generating detailed PDF")
    pdf_detailed = build_detailed_pdf(detail_df, summary_df, "All" if sp_id is None else sps.loc[sps["recordid"] == sp_id, "name"].iloc[0], 
                                     as_of, buckets, selected_customer, grace, length)
    if pdf_detailed is None:
        logger.error("Detailed PDF returned None")
    else:
        with open("detailed_overdues.pdf", "wb") as f:
            f.write(pdf_detailed)
        logger.info("Detailed PDF saved as detailed_overdues.pdf")

if __name__ == "__main__":
    test_overdues()
