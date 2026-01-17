import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- 1. FUNKTIONEN ---

@st.cache_data(ttl=3600)
def get_sp500_tickers_with_sectors():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    df = pd.read_html(response.text)[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return df[['Symbol', 'GICS Sector']]

def calculate_full_details(price, vola, days):
    t = max(days, 1) / 365
    sigma = vola / 100
    strike = price * (1 - (0.65 * sigma * np.sqrt(t)))
    intrinsic = price - strike
    extrinsic = (price * 0.4 * sigma * np.sqrt(t)) * 0.55
    opt_price = intrinsic + extrinsic
    net_debit = price - opt_price
    ann_yield = (extrinsic / max(net_debit, 0.01)) * (365 / max(days, 1)) * 100
    return round(strike, 2), round(opt_price, 2), round(net_debit, 2), round(ann_yield, 1)

# --- 2. OBERFLÄCHE ---
st.set_page_config(page_title="Options Pro 500", layout="wide")
st.title("🎯 S&P 500 Full-Screener (inkl. Vola)")

st.sidebar.header("Filter")
puffer_val = st.sidebar.slider("Mindest Puffer %", 0.0, 15.0, 4.0)
preis_val = st.sidebar.slider("Max Aktienpreis $", 50, 2000, 500)
anzahl_ticker = st.sidebar.selectbox("Scan-Tiefe", [50, 100, 250, 500], index=1)
use_sma = st.sidebar.checkbox("Nur Aufwärtstrend (SMA200)", value=False)
min_vola_filter = st.sidebar.slider("Mindest Vola % für Liste", 10, 50, 20)

# --- 3. LOGIK MIT BATCHING ---

ticker_data = get_sp500_tickers_with_sectors()
selected_tickers = ticker_data[:anzahl_ticker]

if st.button("Scan starten"):
    results = []
    today = datetime.now().date()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_size = 25 
    for start_idx in range(0, len(selected_tickers), batch_size):
        end_idx = start_idx + batch_size
        batch = selected_tickers.iloc[start_idx:end_idx]
        batch_symbols = batch['Symbol'].tolist()
        
        status_text.text(f"Analysiere Batch {start_idx} bis {min(end_idx, len(selected_tickers))}...")
        
        try:
            # Marktdaten Batch-Download
            data_batch = yf.download(batch_symbols, period="2y", interval="1d", group_by='ticker', progress=False, threads=True)
            
            for ticker in batch_symbols:
                try:
                    if ticker not in data_batch.columns.get_level_values(0): continue
                    df = data_batch[ticker].dropna()
                    if len(df) < 200: continue
                    
                    curr_price = float(df['Close'].iloc[-1])
                    vola = np.log(df['Close'] / df['Close'].shift(1)).tail(30).std() * np.sqrt(252) * 100
                    sma200 = df['Close'].rolling(window=200).mean().iloc[-1]

                    # Filter-Abgleich
                    if curr_price > preis_val: continue
                    if use_sma and curr_price < sma200: continue
                    if vola < min_vola_filter: continue

                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    if info.get('profitMargins', 0) < 0.05: continue

                    # Earnings Check
                    days_to_earn = 999
                    cal = stock.calendar
                    if cal is not None and 'Earnings Date' in cal:
                        next_earn = cal['Earnings Date'][0]
                        if hasattr(next_earn, 'date'): next_earn = next_earn.date()
                        days_to_earn = (next_earn - today).days

                    if days_to_earn > 4:
                        trade_dte = 30
                        if days_to_earn < 35: trade_dte = max(5, days_to_earn - 2)
                        
                        strike, opt_price, net_debit, ann_yield = calculate_full_details(curr_price, vola, trade_dte)
                        puffer = round(((curr_price/strike)-1)*100, 1)
                        
                        if puffer >= puffer_val:
                            results.append({
                                'Ticker': ticker,
                                'Preis': round(curr_price, 2),
                                'Vola%': round(vola, 1), # WIEDER DABEI
                                'Laufzeit': trade_dte,
                                'Strike': strike,
                                'Puffer %': puffer,
                                'Rendite p.a.%': ann_yield,
                                'Score': round(ann_yield * (puffer / 10), 2),
                                'Earn in Tg': days_to_earn,
                                'Sektor': batch.loc[batch['Symbol'] == ticker, 'GICS Sector'].values[0],
                                'Beta': info.get('beta', 'N/A'),
                                'RealePrämie$': opt_price,
                                'NetDebit$': net_debit
                            })
                except: continue
        except: continue
        
        progress_bar.progress(min(end_idx / len(selected_tickers), 1.0))

    status_text.empty()
    progress_bar.empty()
    
    df_result = pd.DataFrame(results)
    if not df_result.empty:
        st.success(f"Analyse abgeschlossen! {len(df_result)} Treffer gefunden.")
        # Spalten-Reihenfolge fixieren
        cols = ['Ticker', 'Preis', 'Vola%', 'Laufzeit', 'Strike', 'Puffer %', 'Rendite p.a.%', 'Score', 'Earn in Tg', 'Sektor', 'Beta', 'RealePrämie$', 'NetDebit$']
        st.dataframe(df_result[cols].sort_values(by='Score', ascending=False), use_container_width=True)
    else:
        st.warning("Keine Treffer. Versuche die Filter (z.B. Vola oder Puffer) zu lockern.")
