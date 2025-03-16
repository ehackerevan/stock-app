# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import twstock
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import threading
import logging

app = Flask(__name__)

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# 連接到 SQLite 資料庫
conn = sqlite3.connect('stock_data.db', check_same_thread=False)
cursor = conn.cursor()

# 創建原始資料表（若不存在）
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        date TEXT,
        code TEXT,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (date, code)
    )
''')

# 創建 RS 和 PR 資料表（若不存在）
cursor.execute('''
    CREATE TABLE IF NOT EXISTS rs_pr_data (
        date TEXT,
        code TEXT,
        rs REAL,
        pr REAL,
        PRIMARY KEY (date, code)
    )
''')
conn.commit()

# 全局變數，用於追蹤抓取進度
progress = 0
lock = threading.Lock()

# 取得上市股票代碼
def get_listed_stock_codes():
    try:
        all_stocks = twstock.codes
        listed_stocks = [
            f"{code}.TW" for code, info in all_stocks.items()
            if code.isdigit() and len(code) == 4 and info.type == "股票" and info.market == "上市"
        ]
        listed_stocks.append('^TWII')
        logging.info(f"取得 {len(listed_stocks)-1} 支上市股票及 1 支指數")
        return listed_stocks
    except Exception as e:
        logging.error(f"取得股票代碼失敗: {e}")
        return ['^TWII']

# 取得股票名稱對應表
def get_stock_name_mapping():
    try:
        all_stocks = twstock.codes
        name_mapping = {
            f"{code}.TW": info.name for code, info in all_stocks.items()
            if code.isdigit() and len(code) == 4 and info.type == "股票" and info.market == "上市"
        }
        name_mapping['^TWII'] = '台灣加權指數'
        return name_mapping
    except Exception as e:
        logging.error(f"取得股票名稱失敗: {e}")
        return {}

# 取得資料庫中最新的日期
def get_last_date_in_db(table_name='stock_data'):
    cursor.execute(f"SELECT MAX(date) FROM {table_name}")
    last_date = cursor.fetchone()[0]
    return datetime.strptime(last_date, '%Y-%m-%d').date() if last_date else None

# 合併股票資料
def merge_stock_data(df):
    price_df = df.pivot(index='date', columns='code', values='close')
    volume_df = df.pivot(index='date', columns='code', values='volume')
    return price_df, volume_df

# 計算 IBD RS 值
def calculate_ibd_rs(df):
    rs_dict = {}
    valid_cols = [col for col in df.columns if not df[col].isna().all()]
    days_per_month = 21
    periods = {'3m': 3*days_per_month, '6m': 6*days_per_month, '9m': 9*days_per_month, '12m': 12*days_per_month}
    weights = {'3m': 0.4, '6m': 0.3, '9m': 0.2, '12m': 0.1}
    
    pm_values = pd.Series(index=df.index, dtype=float)
    for date in df.index:
        pm = 0
        for period, period_days in periods.items():
            current_price = df['^TWII'].loc[date]
            past_price = df['^TWII'].shift(period_days).loc[date]
            if pd.notna(current_price) and pd.notna(past_price) and past_price != 0:
                mi = (current_price - past_price) / past_price * 100
                pm += weights[period] * mi
        pm_values.loc[date] = pm
    
    for col in valid_cols:
        if col == '^TWII':
            continue
        ps_values = pd.Series(index=df.index, dtype=float)
        for date in df.index:
            ps = 0
            for period, period_days in periods.items():
                current_price = df[col].loc[date]
                past_price = df[col].shift(period_days).loc[date]
                if pd.notna(current_price) and pd.notna(past_price) and past_price != 0:
                    ri = (current_price - past_price) / past_price * 100
                    ps += weights[period] * ri
            ps_values.loc[date] = ps
        rs_dict[f"{col}_RS"] = ps_values - pm_values
    
    rs_df = pd.concat(rs_dict, axis=1)
    return rs_df

# 計算 PR 值
def calculate_pr(rs_df):
    pr_df = pd.DataFrame(index=rs_df.index)
    pr_columns = [f"{col.replace('_RS', '')}_PR" for col in rs_df.columns]
    pr_df = pr_df.reindex(columns=pr_columns)
    for date in rs_df.index:
        daily_rs = rs_df.loc[date].dropna()
        if not daily_rs.empty:
            pr_values = daily_rs.rank(pct=True) * 99
            for col in rs_df.columns:
                code = col.replace('_RS', '')
                if col in daily_rs.index:
                    pr_df.loc[date, f"{code}_PR"] = pr_values[col]
    return pr_df

# 儲存 RS 和 PR 到資料庫
def store_rs_pr_to_db(rs_df, pr_df):
    for date in rs_df.index:
        for col in rs_df.columns:
            if col.endswith('_RS'):
                code = col.replace('_RS', '')
                rs_value = rs_df.loc[date, col] if pd.notna(rs_df.loc[date, col]) else None
                pr_col = f"{code}_PR"
                pr_value = pr_df.loc[date, pr_col] if pr_col in pr_df.columns and pd.notna(pr_df.loc[date, pr_col]) else None
                if rs_value is not None and pr_value is not None:
                    cursor.execute('''
                        INSERT OR REPLACE INTO rs_pr_data (date, code, rs, pr)
                        VALUES (?, ?, ?, ?)
                    ''', (date.strftime('%Y-%m-%d'), code, rs_value, pr_value))
    conn.commit()
    logging.info("RS 和 PR 值已儲存到資料庫")

# 抓取並儲存資料
def fetch_and_store_data(stock_codes, start_date, end_date):
    global progress
    total = len(stock_codes)
    for i, code in enumerate(stock_codes):
        try:
            df = yf.download(code, start=start_date, end=end_date)
            if df.empty:
                continue
            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            for _, row in df.iterrows():
                date_str = row['Date'] if isinstance(row['Date'], str) else row['Date'].item()
                close_val = float(row['Close']) if isinstance(row['Close'], (int, float)) else float(row['Close'].item())
                volume_val = int(row['Volume']) if isinstance(row['Volume'], (int, float)) else int(row['Volume'].item())
                cursor.execute('''
                    INSERT OR IGNORE INTO stock_data (date, code, close, volume)
                    VALUES (?, ?, ?, ?)
                ''', (date_str, code, close_val, volume_val))
            conn.commit()
            logging.info(f"下載並儲存 {code} 資料成功")
        except Exception as e:
            logging.error(f"下載 {code} 資料時發生錯誤: {e}")
        with lock:
            progress = int((i + 1) / total * 100)
    
    # 計算 RS 和 PR 並儲存
    df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    df['date'] = pd.to_datetime(df['date'])
    price_df, _ = merge_stock_data(df)
    if '^TWII' in price_df.columns:
        rs_df = calculate_ibd_rs(price_df)
        pr_df = calculate_pr(rs_df)
        store_rs_pr_to_db(rs_df, pr_df)

# 從資料庫讀取 RS 和 PR 資料
def load_rs_pr_from_db():
    df = pd.read_sql_query("SELECT * FROM rs_pr_data", conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

# 從資料庫讀取原始資料
def load_stock_data_from_db():
    df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

# 篩選 PR 高於 90 且股價高於 60MA 的股票
def filter_high_pr_stocks(rs_pr_df, stock_df, selected_date, stock_names):
    selected_date = pd.to_datetime(selected_date)
    recent_df = rs_pr_df[rs_pr_df['date'] == selected_date]
    price_df, volume_df = merge_stock_data(stock_df)
    
    ma60_df = price_df.rolling(window=60, min_periods=60).mean()
    
    records = []
    for _, row in recent_df.iterrows():
        if row['pr'] >= 90:
            date = row['date']
            code = row['code']
            volume = volume_df.loc[date, code] if (date in volume_df.index and code in volume_df.columns) else np.nan
            close_price = price_df.loc[date, code] if (date in price_df.index and code in price_df.columns) else np.nan
            ma60 = ma60_df.loc[date, code] if (date in ma60_df.index and code in ma60_df.columns) else np.nan
            
            if (pd.notna(volume) and volume >= 500000 and 
                pd.notna(close_price) and pd.notna(ma60) and close_price > ma60):
                records.append({
                    '日期': date.strftime('%Y-%m-%d'),
                    '公司代號': code,
                    '公司名稱': stock_names.get(code, '未知'),
                    '股價': round(close_price, 2),
                    'PR值': int(round(row['pr'])),
                    '成交量(張)': int(round(volume / 1000))
                })
    return pd.DataFrame(records)

# 篩選 PR 80 以上且 RS 連續上升 3 天的股票
def filter_rising_rs_stocks(rs_pr_df, high_pr_df, selected_date, stock_names):
    if not selected_date:
        return pd.DataFrame(columns=['日期', '公司代號', '公司名稱', '股價', 'PR值'])
    
    selected_date = pd.to_datetime(selected_date)
    start_date = selected_date - timedelta(days=4)
    recent_df = rs_pr_df[(rs_pr_df['date'] >= start_date - timedelta(days=2)) & (rs_pr_df['date'] <= selected_date)]
    excluded_codes = set(high_pr_df['公司代號'].unique()) if not high_pr_df.empty else set()
    price_df, _ = merge_stock_data(load_stock_data_from_db())
    records = []
    
    daily_df = recent_df[recent_df['date'] == selected_date]
    for _, row in daily_df.iterrows():
        if row['pr'] >= 80:
            code = row['code']
            if code in excluded_codes:
                continue
            rs_series = rs_pr_df[rs_pr_df['code'] == code].set_index('date')['rs']
            window_rs = rs_series.loc[selected_date - timedelta(days=2):selected_date]
            close_price = price_df.loc[selected_date, code] if (selected_date in price_df.index and code in price_df.columns) else np.nan
            if len(window_rs) == 3 and all(window_rs.diff().dropna() > 0) and pd.notna(close_price):
                records.append({
                    '日期': selected_date.strftime('%Y-%m-%d'),
                    '公司代號': code,
                    '公司名稱': stock_names.get(code, '未知'),
                    '股價': round(close_price, 2),
                    'PR值': int(round(row['pr']))
                })
    return pd.DataFrame(records)

# 篩選創下 240 天新高的股票（新增 PR 值）
def filter_240_high_stocks(stock_df, rs_pr_df, selected_date, stock_names):
    selected_date = pd.to_datetime(selected_date)
    price_df, volume_df = merge_stock_data(stock_df)
    rs_pr_selected = rs_pr_df[rs_pr_df['date'] == selected_date]
    
    start_date = selected_date - timedelta(days=240)
    recent_price_df = price_df[(price_df.index >= start_date) & (price_df.index <= selected_date)]
    
    records = []
    if selected_date in price_df.index:
        for code in recent_price_df.columns:
            if code == '^TWII':
                continue
            close_price = price_df.loc[selected_date, code] if code in price_df.columns else np.nan
            volume = volume_df.loc[selected_date, code] if (selected_date in volume_df.index and code in volume_df.columns) else np.nan
            past_240_prices = recent_price_df[code].dropna()
            pr_value = rs_pr_selected[rs_pr_selected['code'] == code]['pr'].iloc[0] if not rs_pr_selected[rs_pr_selected['code'] == code].empty else np.nan
            
            if (len(past_240_prices) >= 240 and pd.notna(close_price) and pd.notna(volume) and volume >= 500000 and 
                close_price == past_240_prices.max()):
                records.append({
                    '日期': selected_date.strftime('%Y-%m-%d'),
                    '公司代號': code,
                    '公司名稱': stock_names.get(code, '未知'),
                    '股價': round(close_price, 2),
                    '成交量(張)': int(round(volume / 1000)),
                    'PR值': int(round(pr_value)) if pd.notna(pr_value) else 'N/A'
                })
    return pd.DataFrame(records)

# 取得近 60 天 PR、加權指數和股價資料
@app.route('/get_chart_data/<code>')
def get_chart_data(code):
    rs_pr_df = load_rs_pr_from_db()
    stock_df = load_stock_data_from_db()
    
    if rs_pr_df.empty or stock_df.empty:
        return jsonify({'error': '資料庫中無資料'})
    
    end_date = get_last_date_in_db('rs_pr_data') or datetime.now().date()
    start_date = end_date - timedelta(days=60)
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    rs_pr_df = rs_pr_df[(rs_pr_df['date'] >= start_date) & (rs_pr_df['date'] <= end_date)]
    stock_df = stock_df[(stock_df['date'] >= start_date) & (stock_df['date'] <= end_date)]
    
    company_data = rs_pr_df[rs_pr_df['code'] == code][['date', 'pr']].sort_values('date')
    twii_data = stock_df[stock_df['code'] == '^TWII'][['date', 'close']].sort_values('date')
    stock_price_data = stock_df[stock_df['code'] == code][['date', 'close']].sort_values('date')
    
    dates = company_data['date'].dt.strftime('%Y-%m-%d').tolist()
    pr_values = company_data['pr'].tolist()
    twii_values = twii_data['close'].tolist()
    stock_prices = stock_price_data['close'].tolist()
    
    return jsonify({
        'dates': dates,
        'pr_values': pr_values,
        'twii_values': twii_values,
        'stock_prices': stock_prices
    })

# 主頁路由（單選日期）
@app.route('/', methods=['GET', 'POST'])
def index():
    rs_pr_df = load_rs_pr_from_db()
    stock_df = load_stock_data_from_db()
    stock_names = get_stock_name_mapping()
    
    if rs_pr_df.empty or stock_df.empty:
        recent_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)]
        return render_template('index.html', high_pr=[], rising_rs=[], high_240=[], recent_dates=recent_dates, selected_date=None)
    
    end_date = get_last_date_in_db('rs_pr_data') or datetime.now().date()
    recent_dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)][::-1]
    
    selected_date = request.form.get('date') if request.method == 'POST' else recent_dates[-1]
    
    high_pr_df = filter_high_pr_stocks(rs_pr_df, stock_df, selected_date, stock_names)
    rising_rs_df = filter_rising_rs_stocks(rs_pr_df, high_pr_df, selected_date, stock_names)
    high_240_df = filter_240_high_stocks(stock_df, rs_pr_df, selected_date, stock_names)
    
    return render_template('index.html',
                           high_pr=high_pr_df.to_dict(orient='records'),
                           rising_rs=rising_rs_df.to_dict(orient='records'),
                           high_240=high_240_df.to_dict(orient='records'),
                           recent_dates=recent_dates,
                           selected_date=selected_date)

# 抓取資料路由
@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    global progress
    with lock:
        if progress == 0:
            end_date = datetime.now().date()
            last_date_in_db = get_last_date_in_db('stock_data')
            if last_date_in_db is None or last_date_in_db < end_date - timedelta(days=1):
                start_date = last_date_in_db + timedelta(days=1) if last_date_in_db else (end_date - timedelta(days=3*365))
                stock_codes = get_listed_stock_codes()
                threading.Thread(target=fetch_and_store_data, args=(stock_codes, start_date, end_date)).start()
                return jsonify({'status': 'started'})
            else:
                return jsonify({'status': 'already up-to-date'})
        else:
            return jsonify({'status': 'already running'})

# 進度查詢路由
@app.route('/progress')
def get_progress():
    with lock:
        return jsonify({'progress': progress})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)