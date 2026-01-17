import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- 1. FUNKTIONEN MIT CACHING ---

@st.cache_data(ttl=3600)
def get_sp500_tickers_with_sectors():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        table = pd.read_html(response.text)
        df = table[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df[['Symbol', 'GICS Sector']]
    except:
        return pd.DataFrame(columns=['Symbol', 'GICS Sector'])

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
    return yf.download(tickers, period="1y", interval="1d", group_by='ticker', progress=False)

# --- 2. STREAMLIT OBERFLÄCHE ---

st.set_page_config(page_title="Options Screener Live", layout="wide")
st.title("🎯 Live ITM Covered Call Screener")

# Sidebar Filter
st.sidebar.header("Filter-Einstellungen")
puffer_val = st.sidebar.slider("Mindest Puffer %", 0.0, 15.0, 5.0)
preis_val = st.sidebar.slider("Maximaler Aktienpreis $", 50, 1000, 300)
anzahl_ticker = st.sidebar.selectbox("Anzahl zu prüfender Aktien", [50, 100, 250, 500], index=1)

# --- 3. DATEN-VERARBEITUNG ---

ticker_data = get_sp500_tickers_with_sectors()

if not ticker_data.empty:
    selected_tickers = ticker_data[:anzahl_ticker]
    
    with st.status("Analysiere Markt...", expanded=True) as status:
        st.write("Lade Kurse von Yahoo Finance...")
        data_all = get_market_data(selected_tickers['Symbol'].tolist())
        
        results = []
        today = datetime.now().date()

        for idx, row in selected_tickers.iterrows():
            ticker = row['Symbol']
            try:
                if ticker not in data_all.columns.get_level_values(0): continue
                
                df = data_all[ticker].dropna()
                if len(df) < 150: continue
                
                curr_price = float(df['Close'].iloc[-1])
                if curr_price > preis_val: continue
                
                vola = np.log(df['Close'] / df['Close'].shift(1)).tail(30).std() * np.sqrt(252) * 100
                if vola < 30: continue # Nur Aktien mit Bewegung
                
                sma200 = df['Close'].rolling(window=200).mean().iloc[-1]
                if curr_price < sma200: continue # Nur Aufwärtstrends
                
                # Einzelabfrage für Info/Calendar (hier liegt oft die Verzögerung)
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Fundamentale Stabilität (Margin Filter)
                if info.get('profitMargins', 0) < 0.10: continue

                # Earnings Check
                days_to_earn = 999
                cal = stock.calendar
                if cal is not None and 'Earnings Date' in cal:
                    next_earn = cal['Earnings Date'][0]
                    if hasattr(next_earn, 'date'): next_earn = next_earn.date()
                    days_to_earn = (next_earn - today).days
                
                if days_to_earn > 7:
                    trade_dte = 30
                    if days_to_earn < 35: trade_dte = max(7, days_to_earn - 5)
                    
                    strike, opt_price, net_debit, ann_yield = calculate_full_details(curr_price, vola, trade_dte)
                    puffer = round(((curr_price/strike)-1)*100, 1)
                    
                    if puffer >= puffer_val:
                        score = round(ann_yield * (puffer / 10), 2)
                        results.append({
                            'Ticker': ticker, 'Preis': round(curr_price, 2), 'Vola%': round(vola, 1),
                            'Laufzeit': trade_dte, 'Strike': strike, 'Puffer %': puffer,
                            'Rendite p.a.%': ann_yield, 'Score': score, 'Earn in Tg': days_to_earn,
                            'Sektor': row['GICS Sector'][:12], 'Beta': info.get('beta', 'N/A'),
                            'Support': round(max(sma200, df['Close'].tail(126).min()), 2), 
                            'RealePrämie$': opt_price, 'NetDebit$': net_debit
                        })
                        st.write(f"✅ Treffer gefunden: {ticker}")
            except Exception as e:
                continue # Fehler bei einzelner Aktie ignorieren

        status.update(label="Analyse abgeschlossen!", state="complete", expanded=False)

    df_result = pd.DataFrame(results)

    # --- 4. ANZEIGE ---
    if not df_result.empty:
        st.success(f"{len(df_result)} Kandidaten gefunden.")
        st.dataframe(df_result.sort_values(by='Score', ascending=False), use_container_width=True)
    else:
        st.warning("Keine Treffer. Versuche den Puffer zu verringern (z.B. auf 3%) oder die Anzahl der Aktien zu erhöhen.")
else:
    st.error("Konnte S&P 500 Liste nicht laden. Bitte Seite neu laden.")
