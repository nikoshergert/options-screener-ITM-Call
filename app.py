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

@st.cache_data(ttl=1800)
def get_market_data(tickers):
    # Wir laden etwas mehr Daten für SMA-Berechnung
    return yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False)

# --- 2. OBERFLÄCHE ---

st.set_page_config(page_title="Options Screener Pro", layout="wide")
st.title("🎯 ITM Covered Call Screener Pro")

st.sidebar.header("Strategie-Filter")
puffer_val = st.sidebar.slider("Mindest Puffer %", 0.0, 15.0, 4.0)
preis_val = st.sidebar.slider("Max Aktienpreis $", 50, 1500, 500)
anzahl_ticker = st.sidebar.selectbox("Scan-Tiefe (S&P 500)", [50, 100, 250, 500], index=1)

st.sidebar.subheader("Sicherheits-Optionen")
use_sma = st.sidebar.checkbox("Nur Aufwärtstrend (Preis > SMA200)", value=False)
min_vola = st.sidebar.slider("Mindest Vola %", 10, 50, 25)

# --- 3. LOGIK ---

ticker_data = get_sp500_tickers_with_sectors()
selected_tickers = ticker_data[:anzahl_ticker]

with st.status("Suche nach Kandidaten...", expanded=True) as status:
    data_all = get_market_data(selected_tickers['Symbol'].tolist())
    results = []
    today = datetime.now().date()

    for idx, row in selected_tickers.iterrows():
        ticker = row['Symbol']
        try:
            if ticker not in data_all.columns.get_level_values(0): continue
            df = data_all[ticker].dropna()
            if len(df) < 200: continue
            
            curr_price = float(df['Close'].iloc[-1])
            vola = np.log(df['Close'] / df['Close'].shift(1)).tail(30).std() * np.sqrt(252) * 100
            sma200 = df['Close'].rolling(window=200).mean().iloc[-1]

            # FILTER-LOGIK
            if curr_price > preis_val: continue
            if use_sma and curr_price < sma200: continue
            if vola < min_vola: continue

            # Earnings
            stock = yf.Ticker(ticker)
            days_to_earn = 999
            cal = stock.calendar
            if cal is not None and 'Earnings Date' in cal:
                next_earn = cal['Earnings Date'][0]
                if hasattr(next_earn, 'date'): next_earn = next_earn.date()
                days_to_earn = (next_earn - today).days

            # Nur wenn Earnings mind. 4 Tage weg sind (sehr kurzfristig möglich)
            if days_to_earn > 4:
                trade_dte = 30
                if days_to_earn < 35: trade_dte = max(5, days_to_earn - 2)
                
                strike, opt_price, net_debit, ann_yield = calculate_full_details(curr_price, vola, trade_dte)
                puffer = round(((curr_price/strike)-1)*100, 1)
                
                if puffer >= puffer_val:
                    score = round(ann_yield * (puffer / 10), 2)
                    results.append({
                        'Ticker': ticker, 'Preis': round(curr_price, 2), 'Vola%': round(vola, 1),
                        'Laufzeit': trade_dte, 'Strike': strike, 'Puffer %': puffer,
                        'Rendite p.a.%': ann_yield, 'Score': score, 'Earn in Tg': days_to_earn,
                        'Sektor': row['GICS Sector'], 'Support': round(sma200, 2),
                        'RealePrämie$': opt_price, 'NetDebit$': net_debit
                    })
                    st.write(f"⭐ {ticker} erfüllt alle Kriterien!")
        except: continue

    status.update(label="Analyse fertig!", state="complete", expanded=False)

df_result = pd.DataFrame(results)

if not df_result.empty:
    st.dataframe(df_result.sort_values(by='Score', ascending=False), use_container_width=True)
else:
    st.warning("Keine Treffer gefunden. Tipp: Deaktiviere 'Nur Aufwärtstrend' in der Sidebar oder senke den Puffer.")
