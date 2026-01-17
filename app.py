import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime

# --- KONFIGURATION ---
st.set_page_config(page_title="ITM Pro Screener", layout="wide")

@st.cache_data(ttl=3600)
def get_tickers_pro():
    headers = {'User-Agent': 'Mozilla/5.0'}
    url_sp = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp_sp = requests.get(url_sp, headers=headers)
    sp_table = pd.read_html(io.StringIO(resp_sp.text))[0]
    sp_df = sp_table.rename(columns={'Symbol': 'Symbol', 'GICS Sector': 'Sektor'})[['Symbol', 'Sektor']]

    url_nas = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    resp_nas = requests.get(url_nas, headers=headers)
    nas_tables = pd.read_html(io.StringIO(resp_nas.text))
    nas_df = pd.DataFrame()
    for table in nas_tables:
        cols = [str(c).lower() for c in table.columns]
        if any('ticker' in c or 'symbol' in c for c in cols):
            sym_col = [c for c in table.columns if 'ticker' in c.lower() or 'symbol' in c.lower()][0]
            sec_col = [c for c in table.columns if 'sector' in c.lower()]
            nas_df = table[[sym_col]].rename(columns={sym_col: 'Symbol'})
            nas_df['Sektor'] = table[sec_col[0]] if sec_col else 'Nasdaq Tech'
            break
            
    combined = pd.concat([sp_df, nas_df], ignore_index=True)
    combined['Symbol'] = combined['Symbol'].str.replace('.', '-', regex=False)
    return combined.drop_duplicates(subset=['Symbol'])

def calculate_full_details(price, vola, days):
    t = max(days, 1) / 365
    sigma = vola / 100
    strike = price * (1 - (0.65 * sigma * np.sqrt(t)))
    intrinsic = price - strike
    extrinsic = (price * 0.4 * sigma * np.sqrt(t)) * 0.55
    opt_price = intrinsic + extrinsic
    net_debit = price - opt_price
    ann_yield = (extrinsic / max(net_debit, 0.01)) * (365 / max(days, 1)) * 100
    puffer = ((price/strike)-1)*100
    return round(strike, 2), round(puffer, 1), round(opt_price, 2), round(net_debit, 2), round(ann_yield, 1)

st.title("🎯 ITM Pro Screener (S&P 500 & Nasdaq 100)")

with st.sidebar:
    st.header("Screener Einstellungen")
    max_p = st.number_input("Max. Aktienpreis ($)", 50, 2000, 300)
    min_v = st.slider("Min. Volatilität (%)", 10, 80, 25)
    min_buf = st.slider("Min. ITM Puffer (%)", 1, 20, 5)
    st.markdown("---")
    st.write("**Regeln:**")
    st.write("- Ende immer 2 Tage vor Earnings")
    st.write("- Mindestlaufzeit: 5 Tage")
    st.write("- Trend: Nur über SMA 200")

if st.button("Kombinierten Scan starten"):
    ticker_data = get_tickers_pro()
    results = []
    today = datetime.now().date()
    progress_bar = st.progress(0)
    all_symbols = ticker_data['Symbol'].tolist()
    
    batch_size = 40
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i+batch_size]
        try:
            data = yf.download(batch, period="2y", group_by='ticker', progress=False)
            for ticker in batch:
                try:
                    if ticker not in data.columns.get_level_values(0): continue
                    df_t = data[ticker].dropna()
                    if len(df_t) < 200: continue
                    
                    curr_p = float(df_t['Close'].iloc[-1])
                    if curr_p > max_p: continue
                    
                    sma200 = df_t['Close'].rolling(window=200).mean().iloc[-1]
                    if curr_p < sma200: continue
                    
                    vola = np.log(df_t['Close'] / df_t['Close'].shift(1)).tail(30).std() * np.sqrt(252) * 100
                    if vola < min_v: continue
                    
                    stock = yf.Ticker(ticker)
                    cal = stock.calendar
                    days_to_earn = 999
                    if cal is not None and 'Earnings Date' in cal:
                        next_earn = cal['Earnings Date'][0]
                        if hasattr(next_earn, 'date'): next_earn = next_earn.date()
                        days_to_earn = (next_earn - today).days

                    # --- LOGIK: ENDE VOR EARNINGS (Mindestlaufzeit 5 Tage) ---
                    trade_dte = 30 
                    
                    if days_to_earn < 32:
                        trade_dte = days_to_earn - 2
                    
                    # Wenn Laufzeit unter 5 Tage fällt (wegen nahen Earnings), Ticker überspringen
                    if trade_dte < 5:
                        continue

                    strike, puffer, opt, net, yield_pa = calculate_full_details(curr_p, vola, trade_dte)
                    
                    if puffer >= min_buf:
                        sector = ticker_data.loc[ticker_data['Symbol'] == ticker, 'Sektor'].values[0]
                        support = max(sma200, df_t['Close'].tail(126).min())
                        info = stock.info
                        
                        results.append({
                            'Ticker': ticker, 'Preis': round(curr_p, 2), 'Vola%': round(vola, 1),
                            'Laufzeit': trade_dte, 'Strike': strike, 'Puffer %': puffer,
                            'Rendite p.a.%': yield_pa, 'Score': round(yield_pa * (puffer / 10), 2),
                            'Earn in Tg': days_to_earn, 'Sektor': str(sector)[:15],
                            'Beta': info.get('beta', 'N/A'), 'Support': round(support, 2),
                            'RealePrämie$': opt, 'NetDebit$': net
                        })
                except: continue
        except: continue
        progress_bar.progress(min((i + batch_size) / len(all_symbols), 1.0))

    if results:
        final_df = pd.DataFrame(results).sort_values(by='Score', ascending=False)
        cols = ['Ticker', 'Preis', 'Vola%', 'Laufzeit', 'Strike', 'Puffer %', 'Rendite p.a.%', 
                'Score', 'Earn in Tg', 'Sektor', 'Beta', 'Support', 'RealePrämie$', 'NetDebit$']
        st.dataframe(final_df[cols], use_container_width=True)
        st.success(f"{len(final_df)} Treffer gefunden.")
    else:
        st.warning("Keine Treffer gefunden.")
