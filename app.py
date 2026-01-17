import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- 1. FUNKTIONEN DEFINIEREN ---

def get_sp500_tickers_with_sectors():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    table = pd.read_html(response.text)
    df = table[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return df[['Symbol', 'GICS Sector']]

def calculate_full_details(price, vola, days):
    t = days / 365
    sigma = vola / 100
    strike = price * (1 - (0.65 * sigma * np.sqrt(t)))
    intrinsic = price - strike
    extrinsic = (price * 0.4 * sigma * np.sqrt(t)) * 0.55
    opt_price = intrinsic + extrinsic
    net_debit = price - opt_price
    ann_yield = (extrinsic / net_debit) * (365 / days) * 100
    return round(strike, 2), round(opt_price, 2), round(net_debit, 2), round(ann_yield, 1)

def screen_final_custom(ticker_df, puffer_limit, preis_limit):
    results = []
    today = datetime.now().date()
    tickers = ticker_df['Symbol'].tolist()
    
    # Download der Daten
    data_all = yf.download(tickers, period="1y", interval="1d", group_by='ticker', progress=False)

    for idx, row in ticker_df.iterrows():
        ticker = row['Symbol']
        try:
            df = data_all[ticker].dropna()
            if len(df) < 150: continue
            
            curr_price = float(df['Close'].iloc[-1])
            
            # Preis-Filter
            if curr_price > preis_limit: continue
            
            vola = np.log(df['Close'] / df['Close'].shift(1)).tail(30).std() * np.sqrt(252) * 100
            sma200 = df['Close'].rolling(window=200).mean().iloc[-1]
            support = max(sma200, df['Close'].tail(126).min())

            if curr_price > sma200 and vola > 30:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Earnings Check
                days_to_earn = 999
                cal = stock.calendar
                if cal is not None and 'Earnings Date' in cal:
                    next_earn = cal['Earnings Date'][0].date() if hasattr(cal['Earnings Date'][0], 'date') else cal['Earnings Date'][0]
                    days_to_earn = (next_earn - today).days
                
                if days_to_earn > 7 and info.get('profitMargins', 0) > 0.10:
                    trade_dte = 30
                    if days_to_earn < 35: trade_dte = max(7, days_to_earn - 5)
                    
                    strike, opt_price, net_debit, ann_yield = calculate_full_details(curr_price, vola, trade_dte)
                    puffer = round(((curr_price/strike)-1)*100, 1)
                    
                    # Puffer-Filter
                    if puffer >= puffer_limit:
                        score = round(ann_yield * (puffer / 10), 2)
                        results.append({
                            'Ticker': ticker, 'Preis': round(curr_price, 2), 'Vola%': round(vola, 1),
                            'Laufzeit': trade_dte, 'Strike': strike, 'Puffer %': puffer,
                            'Rendite p.a.%': ann_yield, 'Score': score, 'Earn in Tg': days_to_earn,
                            'Sektor': row['GICS Sector'][:12], 'Beta': info.get('beta', 'N/A'),
                            'Support': round(support, 2), 'RealePrämie$': opt_price, 'NetDebit$': net_debit
                        })
        except: continue
    return pd.DataFrame(results)

# --- 2. STREAMLIT OBERFLÄCHE ---

st.set_page_config(page_title="Options Screener Pro", layout="wide")

st.title("🎯 ITM Covered Call Screener")
st.write("Diese App findet stabile Aktien für deine Cashflow-Strategie.")

# Sidebar für Einstellungen
st.sidebar.header("Filter-Einstellungen")
puffer_val = st.sidebar.slider("Mindest Puffer %", 0.0, 15.0, 5.0)
preis_val = st.sidebar.slider("Maximaler Aktienpreis $", 50, 1000, 300)
anzahl_ticker = st.sidebar.selectbox("Anzahl zu prüfender Aktien", index=1)

if st.button("Markt jetzt scannen"):
    with st.spinner(f"Analysiere die ersten {anzahl_ticker} S&P 500 Aktien..."):
        # Daten holen
        ticker_data = get_sp500_tickers_with_sectors()
        
        # Screening ausführen
        df_result = screen_final_custom(ticker_data[:anzahl_ticker], puffer_val, preis_val)
        
        if not df_result.empty:
            st.success(f"{len(df_result)} Kandidaten gefunden!")
            # Sortierung nach Score
            df_result = df_result.sort_values(by='Score', ascending=False)
            
            # Tabelle anzeigen
            st.dataframe(df_result, use_container_width=True)
            
            # Download Button für Excel/CSV
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Liste als CSV speichern", csv, "options_trades.csv", "text/csv")
        else:
            st.warning("Keine Treffer gefunden. Versuche den Puffer zu verringern oder den Preis zu erhöhen.")

