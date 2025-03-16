from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import twstock
import json

app = Flask(__name__)

# 資料庫連線函數
def get_db_connection():
    conn = sqlite3.connect('stock_data.db')
    return conn

# 從資料庫載入股票數據（優化：限制最近 60 天）
def load_stock_data_from_db(conn):
    df = pd.read_sql_query("SELECT * FROM stock_data WHERE date >= date('now', '-60 days')", conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

# 從資料庫載入 RS/PR 數據（優化：限制最近 60 天）
def load_rs_pr_from_db(conn):
    df = pd.read_sql_query("SELECT * FROM rs_pr_data WHERE date >= date('now', '-60 days')", conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

# 獲取上市股票代碼
def get_listed_stock_codes():
    all_stocks = twstock.codes
    listed_stocks = [
        f"{code}.TW" for code, info in all_stocks.items()
        if code.isdigit() and len(code) == 4 and info.type == "股票" and info.market == "上市"
    ]
    listed_stocks.append('^TWII')
    return listed_stocks

# 獲取股票名稱映射
def get_stock_name_mapping():
    all_stocks = twstock.codes
    name_mapping = {
        f"{code}.TW": info.name for code, info in all_stocks.items()
        if code.isdigit() and len(code) == 4 and info.type == "股票" and info.market == "上市"
    }
    name_mapping['^TWII'] = '台灣加權指數'
    return name_mapping

# 獲取資料庫中最新日期
def get_last_date_in_db(conn, table_name='stock_data'):
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(date) FROM {table_name}")
    last_date = cursor.fetchone()[0]
    return datetime.strptime(last_date, '%Y-%m-%d').date() if last_date else None

# 合併股票數據為價格和成交量 DataFrame
def merge_stock_data(df):
    price_df = df.pivot(index='date', columns='code', values='close')
    volume_df = df.pivot(index='date', columns='code', values='volume')
    return price_df, volume_df

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
    return records

# 篩選 PR 80 以上且 RS 連續上升 3 天的股票
def filter_rising_rs_stocks(rs_pr_df, high_pr_df, selected_date, stock_names, conn):
    selected_date = pd.to_datetime(selected_date)
    start_date = selected_date - timedelta(days=4)
    recent_df = rs_pr_df[(rs_pr_df['date'] >= start_date - timedelta(days=2)) & (rs_pr_df['date'] <= selected_date)]
    excluded_codes = set(item['公司代號'] for item in high_pr_df) if high_pr_df else set()
    price_df, _ = merge_stock_data(load_stock_data_from_db(conn))
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
    return records

# 篩選 240 天新高股票
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
    return records

# 主頁路由
@app.route('/', methods=['GET', 'POST'])
def index():
    conn = get_db_connection()
    rs_pr_df = load_rs_pr_from_db(conn)
    stock_df = load_stock_data_from_db(conn)
    stock_names = get_stock_name_mapping()
    
    end_date = get_last_date_in_db(conn, 'rs_pr_data') or datetime.now().date()
    recent_dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)][::-1]
    selected_date = request.form.get('date', recent_dates[-1])
    
    high_pr_df = filter_high_pr_stocks(rs_pr_df, stock_df, selected_date, stock_names)
    rising_rs_df = filter_rising_rs_stocks(rs_pr_df, high_pr_df, selected_date, stock_names, conn)
    high_240_df = filter_240_high_stocks(stock_df, rs_pr_df, selected_date, stock_names)
    
    conn.close()
    return render_template('index.html', high_pr=high_pr_df, rising_rs=rising_rs_df, high_240=high_240_df, recent_dates=recent_dates, selected_date=selected_date)

# 獲取圖表數據路由
@app.route('/get_chart_data/<code>')
def get_chart_data(code):
    conn = get_db_connection()
    rs_pr_df = load_rs_pr_from_db(conn)
    stock_df = load_stock_data_from_db(conn)
    
    end_date = get_last_date_in_db(conn, 'rs_pr_data') or datetime.now().date()
    start_date = end_date - timedelta(days=60)
    
    company_data = rs_pr_df[rs_pr_df['code'] == code][['date', 'pr']].sort_values('date')
    twii_data = stock_df[stock_df['code'] == '^TWII'][['date', 'close']].sort_values('date')
    stock_price_data = stock_df[stock_df['code'] == code][['date', 'close']].sort_values('date')
    
    dates = company_data['date'].dt.strftime('%Y-%m-%d').tolist()
    pr_values = company_data['pr'].tolist()
    twii_values = twii_data['close'].tolist()
    stock_prices = stock_price_data['close'].tolist()
    
    conn.close()
    return json.dumps({
        'dates': dates,
        'pr_values': pr_values,
        'twii_values': twii_values,
        'stock_prices': stock_prices
    })

# 抓取最新數據路由（本地測試用，可選）
@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    conn = get_db_connection()
    codes = get_listed_stock_codes()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    stock_data = yf.download(codes, start=start_date, end=end_date)
    stock_df = stock_data.stack().reset_index()
    stock_df.columns = ['date', 'code', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
    stock_df.to_sql('stock_data', conn, if_exists='replace', index=False)
    
    # 這裡應添加 RS/PR 計算邏輯，簡化為示例
    rs_pr_df = stock_df.groupby('code').apply(lambda x: x.tail(60)).reset_index(drop=True)
    rs_pr_df['date'] = pd.to_datetime(rs_pr_df['date'])
    rs_pr_df['pr'] = np.random.randint(0, 100, size=len(rs_pr_df))  # 模擬 PR 值
    rs_pr_df['rs'] = np.random.randint(0, 100, size=len(rs_pr_df))  # 模擬 RS 值
    rs_pr_df.to_sql('rs_pr_data', conn, if_exists='replace', index=False)
    
    conn.close()
    return '', 204

if __name__ == '__main__':
    # 本地測試用，Render 上不執行
    import os
    port = int(os.getenv("PORT", 5000))  # 默認 5000，若有 PORT 則使用
    app.run(host='0.0.0.0', port=port, debug=True)