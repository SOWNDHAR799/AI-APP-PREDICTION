import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import warnings
import yfinance as yf
import requests
import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Market Predictor Pro", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

# ── Complete Groww-Level Stock Map (250+ Stocks) ──────────────────────────
STOCK_MAP = {
    # ═══ INDICES ═══
    'NIFTY': '^NSEI', 'NIFTY50': '^NSEI', 'NSE': '^NSEI',
    'SENSEX': '^BSESN', 'BSE': '^BSESN',
    'BANKNIFTY': '^NSEBANK', 'NIFTYBANK': '^NSEBANK',
    'MIDCPNIFTY': '^NSEMDCP50', 'FINNIFTY': '^CNXFIN',
    'NIFTYIT': '^CNXIT', 'NIFTYPHARMA': '^CNXPHARMA',
    'NIFTYAUTO': '^CNXAUTO', 'NIFTYMETAL': '^CNXMETAL',
    'NIFTYREALTY': '^CNXREALTY', 'NIFTYFMCG': '^CNXFMCG',
    'NIFTYENERGY': '^CNXENERGY',

    # ═══ NIFTY 50 (All 50 stocks) ═══
    'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'INFY': 'INFY.NS',
    'HDFCBANK': 'HDFCBANK.NS', 'HDFC': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS', 'ICICI': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS', 'BHARTIARTL': 'BHARTIARTL.NS',
    'ITC': 'ITC.NS', 'KOTAKBANK': 'KOTAKBANK.NS',
    'LT': 'LT.NS', 'AXISBANK': 'AXISBANK.NS',
    'WIPRO': 'WIPRO.NS', 'MARUTI': 'MARUTI.NS',
    'SUNPHARMA': 'SUNPHARMA.NS', 'BAJFINANCE': 'BAJFINANCE.NS',
    'BAJFINSV': 'BAJFINSV.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'HCLTECH': 'HCLTECH.NS', 'ASIANPAINT': 'ASIANPAINT.NS',
    'TITAN': 'TITAN.NS', 'ULTRACEMCO': 'ULTRACEMCO.NS',
    'NESTLEIND': 'NESTLEIND.NS', 'TECHM': 'TECHM.NS',
    'POWERGRID': 'POWERGRID.NS', 'NTPC': 'NTPC.NS',
    'ONGC': 'ONGC.NS', 'COALINDIA': 'COALINDIA.NS',
    'DRREDDY': 'DRREDDY.NS', 'CIPLA': 'CIPLA.NS',
    'DIVISLAB': 'DIVISLAB.NS', 'EICHERMOT': 'EICHERMOT.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS', 'M&M': 'M&M.NS',
    'ADANIENT': 'ADANIENT.NS', 'ADANIPORTS': 'ADANIPORTS.NS',
    'JSWSTEEL': 'JSWSTEEL.NS', 'HINDALCO': 'HINDALCO.NS',
    'BPCL': 'BPCL.NS', 'HINDUNILVR': 'HINDUNILVR.NS',
    'BRITANNIA': 'BRITANNIA.NS', 'INDUSINDBK': 'INDUSINDBK.NS',
    'TATASTEEL': 'TATASTEEL.NS', 'TATA': 'TATASTEEL.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'GRASIM': 'GRASIM.NS', 'APOLLOHOSP': 'APOLLOHOSP.NS',
    'SBILIFE': 'SBILIFE.NS', 'HDFCLIFE': 'HDFCLIFE.NS',
    'LTIM': 'LTIM.NS', 'LTTS': 'LTTS.NS',

    # ═══ TATA GROUP (Complete) ═══
    'TATAPOWER': 'TATAPOWER.NS', 'TATACHEM': 'TATACHEM.NS',
    'TATACOMM': 'TATACOMM.NS', 'TATAELXSI': 'TATAELXSI.NS',
    'TATAINVEST': 'TATAINVEST.NS', 'TATACONSUM': 'TATACONSUM.NS',
    'TATAMETALI': 'TATAMETALI.NS', 'TATASPONGE': 'TATASPONGE.NS',
    'TATACOFFEE': 'TATACOFFEE.NS', 'TTML': 'TTML.NS',
    'TITAN': 'TITAN.NS', 'VOLTAS': 'VOLTAS.NS',
    'TRENT': 'TRENT.NS', 'RALLIS': 'RALLIS.NS',

    # ═══ ADANI GROUP (Complete) ═══
    'ADANIGREEN': 'ADANIGREEN.NS', 'ADANIPOWER': 'ADANIPOWER.NS',
    'ADANITRANS': 'ADANITRANS.NS', 'ATGL': 'ATGL.NS',
    'AWL': 'AWL.NS', 'ADANIWILMAR': 'AWL.NS',
    'ACC': 'ACC.NS', 'AMBUJACEM': 'AMBUJACEM.NS',

    # ═══ BANKING & FINANCE ═══
    'PNB': 'PNB.NS', 'BANKBARODA': 'BANKBARODA.NS',
    'CANBK': 'CANBK.NS', 'UNIONBANK': 'UNIONBANK.NS',
    'IDFCFIRSTB': 'IDFCFIRSTB.NS', 'FEDERALBNK': 'FEDERALBNK.NS',
    'BANDHANBNK': 'BANDHANBNK.NS', 'RBLBANK': 'RBLBANK.NS',
    'YESBANK': 'YESBANK.NS', 'AUBANK': 'AUBANK.NS',
    'CENTRALBK': 'CENTRALBK.NS', 'INDIANB': 'INDIANB.NS',
    'IOB': 'IOB.NS', 'UCOBANK': 'UCOBANK.NS',
    'MAHABANK': 'MAHABANK.NS', 'PSB': 'PSB.NS',
    'J&KBANK': 'J&KBANK.NS', 'KARURVYSYA': 'KARURVYSYA.NS',
    'SOUTHBANK': 'SOUTHBANK.NS', 'TMB': 'TMB.NS',
    'CUB': 'CUB.NS', 'DCB': 'DCB.NS', 'CSB': 'CSB.NS',

    # ═══ NBFC & FINANCE ═══
    'SHRIRAMFIN': 'SHRIRAMFIN.NS', 'CHOLAFIN': 'CHOLAFIN.NS',
    'MUTHOOTFIN': 'MUTHOOTFIN.NS', 'MANAPPURAM': 'MANAPPURAM.NS',
    'CANFINHOME': 'CANFINHOME.NS', 'LICHSGFIN': 'LICHSGFIN.NS',
    'POONAWALLA': 'POONAWALLA.NS', 'IIFL': 'IIFL.NS',
    'MFSL': 'MFSL.NS', 'MOTILALOFS': 'MOTILALOFS.NS',
    'ANGELONE': 'ANGELONE.NS', 'CDSL': 'CDSL.NS',
    'BSE': 'BSE.NS', 'MCX': 'MCX.NS',
    'ICICIPRULI': 'ICICIPRULI.NS', 'ICICIGI': 'ICICIGI.NS',
    'LICI': 'LICI.NS', 'NIACL': 'NIACL.NS',
    'GICRE': 'GICRE.NS', 'STARHEALTH': 'STARHEALTH.NS',
    'POLICYBZR': 'POLICYBZR.NS',

    # ═══ IT & SOFTWARE ═══
    'MPHASIS': 'MPHASIS.NS', 'COFORGE': 'COFORGE.NS',
    'PERSISTENT': 'PERSISTENT.NS', 'HAPPSTMNDS': 'HAPPSTMNDS.NS',
    'ZOMATO': 'ZOMATO.NS', 'PAYTM': 'PAYTM.NS',
    'NAUKRI': 'NAUKRI.NS', 'INFOEDGE': 'NAUKRI.NS',
    'ROUTE': 'ROUTE.NS', 'MAPMY': 'MAPMYINDIA.NS',
    'LATENTVIEW': 'LATENTVIEW.NS', 'NEWGEN': 'NEWGEN.NS',
    'MASTEK': 'MASTEK.NS', 'ZENSAR': 'ZENSAR.NS',
    'BIRLASOFT': 'BIRLASOFT.NS', 'CYIENT': 'CYIENT.NS',
    'KPITTECH': 'KPITTECH.NS', 'SONATSOFTW': 'SONATSOFTW.NS',
    'TANLA': 'TANLA.NS', 'RATEGAIN': 'RATEGAIN.NS',

    # ═══ PHARMA & HEALTHCARE ═══
    'LUPIN': 'LUPIN.NS', 'AUROPHARMA': 'AUROPHARMA.NS',
    'BIOCON': 'BIOCON.NS', 'TORNTPHARM': 'TORNTPHARM.NS',
    'ALKEM': 'ALKEM.NS', 'IPCALAB': 'IPCALAB.NS',
    'GLENMARK': 'GLENMARK.NS', 'ABBOTINDIA': 'ABBOTINDIA.NS',
    'PFIZER': 'PFIZER.NS', 'SANOFI': 'SANOFI.NS',
    'LALPATHLAB': 'LALPATHLAB.NS', 'METROPOLIS': 'METROPOLIS.NS',
    'MAXHEALTH': 'MAXHEALTH.NS', 'FORTIS': 'FORTIS.NS',
    'APOLLOHOSP': 'APOLLOHOSP.NS', 'NARAYANA': 'NH.NS',
    'NATCOPHARM': 'NATCOPHARM.NS', 'LAURUSLABS': 'LAURUSLABS.NS',
    'GRANULES': 'GRANULES.NS', 'AJANTPHARM': 'AJANTPHARM.NS',

    # ═══ AUTO & AUTO ANCILLARY ═══
    'ASHOKLEY': 'ASHOKLEY.NS', 'TVSMOTOR': 'TVSMOTOR.NS',
    'BALKRISIND': 'BALKRISIND.NS', 'MRF': 'MRF.NS',
    'APOLLOTYRE': 'APOLLOTYRE.NS', 'CEATLTD': 'CEATLTD.NS',
    'BHARATFORG': 'BHARATFORG.NS', 'MOTHERSON': 'MOTHERSON.NS',
    'EXIDEIND': 'EXIDEIND.NS', 'AMARAJABAT': 'AMARARAJA.NS',
    'BOSCHLTD': 'BOSCHLTD.NS', 'ENDURANCE': 'ENDURANCE.NS',
    'SUNDRMFAST': 'SUNDRMFAST.NS', 'ESCORTS': 'ESCORTS.NS',
    'OLACABS': 'OLACABS.NS', 'ABORANGE': 'OLA.NS',

    # ═══ METALS & MINING ═══
    'VEDL': 'VEDL.NS', 'VEDANTA': 'VEDL.NS',
    'NMDC': 'NMDC.NS', 'NATIONALUM': 'NATIONALUM.NS',
    'SAIL': 'SAIL.NS', 'JINDALSTEL': 'JINDALSTEL.NS',
    'JSWENERGY': 'JSWENERGY.NS', 'JSWINFRA': 'JSWINFRA.NS',
    'RATNAMANI': 'RATNAMANI.NS', 'WELCORP': 'WELCORP.NS',
    'APLAPOLLO': 'APLAPOLLO.NS', 'GALLANTT': 'GALLANTT.NS',

    # ═══ OIL, GAS & ENERGY ═══
    'IOC': 'IOC.NS', 'GAIL': 'GAIL.NS',
    'PETRONET': 'PETRONET.NS', 'HINDPETRO': 'HINDPETRO.NS',
    'MGL': 'MGL.NS', 'IGL': 'IGL.NS', 'GUJGASLTD': 'GUJGASLTD.NS',
    'TATAPOWER': 'TATAPOWER.NS', 'TORNTPOWER': 'TORNTPOWER.NS',
    'CESC': 'CESC.NS', 'NHPC': 'NHPC.NS', 'SJVN': 'SJVN.NS',
    'IREDA': 'IREDA.NS', 'RECLTD': 'RECLTD.NS', 'PFC': 'PFC.NS',

    # ═══ FMCG & CONSUMER ═══
    'DABUR': 'DABUR.NS', 'GODREJCP': 'GODREJCP.NS',
    'MARICO': 'MARICO.NS', 'COLPAL': 'COLPAL.NS',
    'EMAMILTD': 'EMAMILTD.NS', 'GILLETTE': 'GILLETTE.NS',
    'PGHH': 'PGHH.NS', 'JYOTHYLAB': 'JYOTHYLAB.NS',
    'VGUARD': 'VGUARD.NS', 'BATAINDIA': 'BATAINDIA.NS',
    'RELAXO': 'RELAXO.NS', 'PAGEIND': 'PAGEIND.NS',
    'TRENT': 'TRENT.NS', 'DMART': 'DMART.NS',
    'DEVYANI': 'DEVYANI.NS', 'JUBLFOOD': 'JUBLFOOD.NS',
    'ZOMATO': 'ZOMATO.NS', 'SWIGGY': 'SWIGGY.NS',
    'PATANJALI': 'PATANJALI.NS',

    # ═══ CEMENT & CONSTRUCTION ═══
    'ULTRACEMCO': 'ULTRACEMCO.NS', 'SHREECEM': 'SHREECEM.NS',
    'AMBUJACEM': 'AMBUJACEM.NS', 'ACC': 'ACC.NS',
    'DALMIACMNT': 'DALMIACMNT.NS', 'RAMCOCEM': 'RAMCOCEM.NS',
    'JKCEMENT': 'JKCEMENT.NS', 'BIRLACEM': 'BIRLACEM.NS',
    'JKLAKSHMI': 'JKLAKSHMI.NS',

    # ═══ REAL ESTATE ═══
    'DLF': 'DLF.NS', 'GODREJPROP': 'GODREJPROP.NS',
    'OBEROIRLTY': 'OBEROIRLTY.NS', 'PRESTIGE': 'PRESTIGE.NS',
    'PHOENIXLTD': 'PHOENIXLTD.NS', 'BRIGADE': 'BRIGADE.NS',
    'SOBHA': 'SOBHA.NS', 'LODHA': 'LODHA.NS',
    'MAHLIFE': 'MAHLIFE.NS', 'SUNTECK': 'SUNTECK.NS',
    'RAYMOND': 'RAYMOND.NS',

    # ═══ INDUSTRIAL & ENGINEERING ═══
    'SIEMENS': 'SIEMENS.NS', 'ABB': 'ABB.NS',
    'CUMMINSIND': 'CUMMINSIND.NS', 'HAVELLS': 'HAVELLS.NS',
    'CROMPTON': 'CROMPTON.NS', 'BLUESTARLT': 'BLUESTARLT.NS',
    'POLYCAB': 'POLYCAB.NS', 'KEI': 'KEI.NS',
    'AFFLE': 'AFFLE.NS', 'DIXON': 'DIXON.NS',
    'KAYNES': 'KAYNES.NS', 'ELGIEQUIP': 'ELGIEQUIP.NS',
    'THERMAX': 'THERMAX.NS', 'GRINDWELL': 'GRINDWELL.NS',
    'CARBORUNIV': 'CARBORUNIV.NS', 'BEL': 'BEL.NS',
    'HAL': 'HAL.NS', 'BDL': 'BDL.NS', 'MAZAGON': 'MAZDOCK.NS',
    'COCHINSHIP': 'COCHINSHIP.NS', 'GRSE': 'GRSE.NS',

    # ═══ TELECOM & MEDIA ═══
    'BHARTIARTL': 'BHARTIARTL.NS', 'IDEA': 'IDEA.NS',
    'TTML': 'TTML.NS', 'HATHWAY': 'HATHWAY.NS',
    'DEN': 'DEN.NS', 'SUNTV': 'SUNTV.NS',
    'ZEEL': 'ZEEL.NS', 'PVR': 'PVRINOX.NS',
    'SAREGAMA': 'SAREGAMA.NS', 'NAZARA': 'NAZARA.NS',

    # ═══ RAILWAYS & INFRA ═══
    'IRCTC': 'IRCTC.NS', 'IRFC': 'IRFC.NS',
    'RVNL': 'RVNL.NS', 'RAILTEL': 'RAILTEL.NS',
    'RITES': 'RITES.NS', 'TITAGARH': 'TITAGARH.NS',
    'TEXRAIL': 'TEXRAIL.NS',
    'IRB': 'IRB.NS', 'KNR': 'KNRCON.NS',
    'NBCC': 'NBCC.NS', 'NCC': 'NCC.NS',
    'HCC': 'HCC.NS', 'ASHOKA': 'ASHOKA.NS',

    # ═══ CHEMICAL & FERTILIZER ═══
    'PIDILITIND': 'PIDILITIND.NS', 'UPL': 'UPL.NS',
    'AARTI': 'AARTIIND.NS', 'SRF': 'SRF.NS',
    'DEEPAKNTR': 'DEEPAKNTR.NS', 'NAVINFLUOR': 'NAVINFLUOR.NS',
    'PIIND': 'PIIND.NS', 'CLEAN': 'CLEAN.NS',
    'FLUOROCHEM': 'FLUOROCHEM.NS', 'ALKYLAMINE': 'ALKYLAMINE.NS',
    'CHAMBALFERT': 'CHAMBLFERT.NS', 'COROMANDEL': 'COROMANDEL.NS',
    'GNFC': 'GNFC.NS', 'NFL': 'NFL.NS',
    'RCF': 'RCF.NS', 'FACT': 'FACT.NS',

    # ═══ TEXTILES & APPAREL ═══
    'ARVIND': 'ARVIND.NS', 'RAYMOND': 'RAYMOND.NS',
    'TRIDENT': 'TRIDENT.NS', 'WELSPUNLIV': 'WELSPUNLIV.NS',
    'KPRMILL': 'KPRMILL.NS', 'GOKALDAS': 'GOKALDAS.NS',

    # ═══ NIPPON / AMC / MUTUAL FUND ═══
    'NIPPONLIFE': 'NAM-INDIA.NS', 'NAM-INDIA': 'NAM-INDIA.NS',
    'NIPPON': 'NAM-INDIA.NS',
    'HDFCAMC': 'HDFCAMC.NS', 'UTIAMC': 'UTIAMC.NS',

    # ═══ PSU & GOVERNMENT ═══
    'IRCTC': 'IRCTC.NS', 'COALINDIA': 'COALINDIA.NS',
    'NHPC': 'NHPC.NS', 'BEL': 'BEL.NS', 'HAL': 'HAL.NS',
    'CONCOR': 'CONCOR.NS', 'HUDCO': 'HUDCO.NS',
    'NMDC': 'NMDC.NS', 'NLCINDIA': 'NLCINDIA.NS',
    'SAIL': 'SAIL.NS', 'BHEL': 'BHEL.NS',
    'OFSS': 'OFSS.NS', 'COCHINSHIP': 'COCHINSHIP.NS',

    # ═══ ETFs (Popular on Groww) ═══
    'GOLDBEES': 'GOLDBEES.NS', 'SILVERBEES': 'SILVERBEES.NS',
    'NIFTYBEES': 'NIFTYBEES.NS', 'BANKBEES': 'BANKBEES.NS',
    'JUNIORBEES': 'JUNIORBEES.NS', 'LIQUIDBEES': 'LIQUIDBEES.NS',
    'SETFNIF50': 'SETFNIF50.NS', 'ITETF': 'ITETF.NS',
    'NIPPON GOLD ETF': 'GOLDBEES.NS', 'NIPPON SILVER ETF': 'SILVERBEES.NS',
    'TATASILV': 'TATSILV.NS', 'TATA SILVER': 'TATSILV.NS', 'TATASILVER': 'TATSILV.NS',
    'TATAGOLD': 'TATAGOLD.NS', 'TATA GOLD': 'TATAGOLD.NS',
    'TATA MOTORS': 'TATAMOTORS.NS', 'TATA MOTORE': 'TATAMOTORS.NS', 'TATA MOTORES': 'TATAMOTORS.NS',
    'TATA STEEL': 'TATASTEEL.NS', 'TATA POWER': 'TATAPOWER.NS',
    'TATA CHEM': 'TATACHEM.NS', 'TATA ELXSI': 'TATAELXSI.NS',
    'TATA CONSUMER': 'TATACONSUM.NS', 'TATA COMM': 'TATACOMM.NS',

    # ═══ NEW-AGE TECH / STARTUPS ═══
    'ZOMATO': 'ZOMATO.NS', 'PAYTM': 'PAYTM.NS',
    'NYKAA': 'NYKAA.NS', 'POLICYBZR': 'POLICYBZR.NS',
    'CARTRADE': 'CARTRADE.NS', 'DELHIVERY': 'DELHIVERY.NS',
    'MAPMYINDIA': 'MAPMYINDIA.NS',

    # ═══ COMMODITIES ═══
    'GOLD': 'GC=F', 'SILVER': 'SI=F', 'CRUDE': 'CL=F', 'CRUDEOIL': 'CL=F',
    'NATURALGAS': 'NG=F', 'COPPER': 'HG=F',

    # ═══ US STOCKS (Popular on Groww) ═══
    'AAPL': 'AAPL', 'GOOGL': 'GOOGL', 'MSFT': 'MSFT', 'AMZN': 'AMZN',
    'TSLA': 'TSLA', 'NVDA': 'NVDA', 'META': 'META', 'NFLX': 'NFLX',
    'AMD': 'AMD', 'INTC': 'INTC', 'CRM': 'CRM', 'ORCL': 'ORCL',
    'UBER': 'UBER', 'SNAP': 'SNAP', 'COIN': 'COIN', 'PLTR': 'PLTR',
    'DIS': 'DIS', 'BA': 'BA', 'JPM': 'JPM', 'V': 'V', 'MA': 'MA',
    'KO': 'KO', 'PEP': 'PEP', 'WMT': 'WMT', 'PG': 'PG', 'JNJ': 'JNJ',
}

COMMODITY_USD = {'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F'}

# ── Groww-Style Categories (20+ Sectors) ──────────────────────────────────
DASHBOARD_CATEGORIES = {
    '🏛️ Indices': ['NIFTY', 'SENSEX', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY'],
    '🔵 Tata Group': ['TATASTEEL', 'TATAMOTORS', 'TATAPOWER', 'TATACONSUM', 'TATAELXSI',
                       'TATACOMM', 'TATACHEM', 'TATAINVEST', 'TITAN', 'VOLTAS', 'TRENT', 'RALLIS'],
    '🏢 Adani Group': ['ADANIENT', 'ADANIPORTS', 'ADANIGREEN', 'ADANIPOWER', 'ATGL', 'AWL', 'ACC', 'AMBUJACEM'],
    '🏦 Public Banks': ['SBIN', 'PNB', 'BANKBARODA', 'CANBK', 'UNIONBANK', 'IOB', 'CENTRALB', 'INDIANB', 'UCOBANK', 'MAHABANK', 'PSB'],
    '🏧 Private Banks': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
                          'IDFCFIRSTB', 'FEDERALBNK', 'BANDHANBNK', 'RBLBANK', 'YESBANK', 'AUBANK', 'CUB', 'CSB', 'TMB', 'KARURVYSYA'],
    '💰 NBFC & Finance': ['BAJFINANCE', 'BAJFINSV', 'SHRIRAMFIN', 'CHOLAFIN', 'MUTHOOTFIN',
                           'MANAPPURAM', 'LICHSGFIN', 'CANFINHOME', 'POONAWALLA', 'IIFL', 'MFSL', 'MOTILALOFS', 'ANGELONE'],
    '🛡️ Insurance': ['LICI', 'SBILIFE', 'HDFCLIFE', 'ICICIPRULI', 'ICICIGI', 'STARHEALTH', 'NIACL', 'GICRE', 'POLICYBZR'],
    '💻 IT & Software': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'LTTS',
                          'MPHASIS', 'COFORGE', 'PERSISTENT', 'HAPPSTMNDS', 'KPITTECH',
                          'MASTEK', 'ZENSAR', 'BIRLASOFT', 'CYIENT', 'SONATSOFTW', 'TATAELXSI'],
    '📱 New-Age Tech': ['ZOMATO', 'PAYTM', 'NYKAA', 'POLICYBZR', 'DELHIVERY', 'CARTRADE', 'MAPMYINDIA', 'NAZARA'],
    '💊 Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN', 'AUROPHARMA',
                   'BIOCON', 'TORNTPHARM', 'ALKEM', 'IPCALAB', 'GLENMARK', 'NATCOPHARM',
                   'LAURUSLABS', 'GRANULES', 'AJANTPHARM', 'ABBOTINDIA', 'PFIZER', 'SANOFI'],
    '🏥 Healthcare': ['APOLLOHOSP', 'MAXHEALTH', 'FORTIS', 'LALPATHLAB', 'METROPOLIS'],
    '🚗 Auto': ['TATAMOTORS', 'MARUTI', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT',
                'ASHOKLEY', 'TVSMOTOR', 'ESCORTS'],
    '🔧 Auto Ancillary': ['BOSCHLTD', 'MOTHERSON', 'BHARATFORG', 'BALKRISIND', 'MRF',
                           'APOLLOTYRE', 'CEATLTD', 'EXIDEIND', 'ENDURANCE', 'SUNDRMFAST'],
    '🏭 Metal & Mining': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'NMDC', 'NATIONALUM',
                           'SAIL', 'JINDALSTEL', 'COALINDIA', 'APLAPOLLO', 'RATNAMANI', 'WELCORP', 'GALLANTT'],
    '⛽ Oil & Gas': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'HINDPETRO', 'GAIL',
                     'PETRONET', 'MGL', 'IGL', 'GUJGASLTD'],
    '⚡ Power & Energy': ['NTPC', 'POWERGRID', 'TATAPOWER', 'ADANIGREEN', 'ADANIPOWER',
                          'TORNTPOWER', 'CESC', 'NHPC', 'SJVN', 'IREDA', 'RECLTD', 'PFC', 'JSWENERGY'],
    '🏗️ Cement': ['ULTRACEMCO', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALMIACMNT', 'RAMCOCEM', 'JKCEMENT', 'JKLAKSHMI'],
    '🏠 Real Estate': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'PHOENIXLTD', 'BRIGADE', 'SOBHA', 'LODHA', 'SUNTECK'],
    '🛒 FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP',
                 'MARICO', 'COLPAL', 'TATACONSUM', 'EMAMILTD', 'PATANJALI', 'JUBLFOOD', 'DMART'],
    '🏭 Industrial & Engineering': ['LT', 'SIEMENS', 'ABB', 'HAVELLS', 'CROMPTON', 'POLYCAB',
                                     'KEI', 'DIXON', 'THERMAX', 'CUMMINSIND', 'ELGIEQUIP', 'GRINDWELL'],
    '🛡️ Defence': ['HAL', 'BEL', 'BDL', 'MAZAGON', 'COCHINSHIP', 'GRSE'],
    '🚂 Railways': ['IRCTC', 'IRFC', 'RVNL', 'RAILTEL', 'RITES', 'TITAGARH'],
    '📡 Telecom': ['BHARTIARTL', 'IDEA', 'TTML'],
    '🧪 Chemicals': ['PIDILITIND', 'UPL', 'SRF', 'DEEPAKNTR', 'NAVINFLUOR', 'PIIND',
                      'FLUOROCHEM', 'AARTI', 'CLEAN', 'ALKYLAMINE'],
    '🌾 Fertilizer': ['CHAMBALFERT', 'COROMANDEL', 'GNFC', 'NFL', 'RCF', 'FACT'],
    '👔 Textiles': ['ARVIND', 'PAGEIND', 'TRENT', 'RAYMOND', 'TRIDENT', 'WELSPUNLIV', 'KPRMILL', 'GOKALDAS'],
    '📈 AMC & Exchange': ['NIPPON', 'HDFCAMC', 'UTIAMC', 'CDSL', 'MCX', 'ANGELONE'],
    '🏆 Commodities': ['GOLD', 'SILVER', 'CRUDE', 'NATURALGAS', 'COPPER'],
    '📦 ETFs': ['GOLDBEES', 'SILVERBEES', 'TATASILV', 'TATAGOLD', 'NIFTYBEES', 'BANKBEES', 'JUNIORBEES'],
    '🇺🇸 US Stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                       'AMD', 'INTC', 'CRM', 'ORCL', 'UBER', 'PLTR', 'DIS', 'BA',
                       'JPM', 'V', 'MA', 'KO', 'PEP', 'WMT', 'PG', 'JNJ'],
}

WATCHLIST_DEFAULT = ['WIPRO', 'RELIANCE', 'TATASTEEL', 'NIPPON', 'BOSCHLTD', 'VEDL',
                     'TATAMOTORS', 'GOLD', 'SILVER', 'ADANIGREEN', 'SHRIRAMFIN', 'CHOLAFIN',
                     'IRCTC', 'HAL', 'ZOMATO', 'DLF']

# ── Premium CSS (Groww-inspired) ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Ticker Bar */
.ticker-bar {
    background: #0f172a; padding: 8px 16px; border-radius: 8px; margin-bottom: 1rem;
    display: flex; gap: 24px; overflow-x: auto; white-space: nowrap; font-size: 0.85rem;
    border: 1px solid #1e293b;
}
.ticker-item { display: inline-block; }
.ticker-name { color: #94a3b8; font-weight: 600; }
.ticker-price { color: #e2e8f0; font-weight: 700; margin-left: 6px; }
.ticker-up { color: #10b981; font-weight: 600; margin-left: 4px; }
.ticker-down { color: #ef4444; font-weight: 600; margin-left: 4px; }

/* Main Title */
.main-title {
    font-size: 2rem; font-weight: 800; text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.sub-title { text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }

/* Stock Cards (Groww style) */
.stock-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 1rem; margin: 0.4rem 0; transition: all 0.2s;
    cursor: pointer;
}
.stock-card:hover { border-color: #667eea; transform: translateY(-2px); box-shadow: 0 4px 20px rgba(102,126,234,0.15); }
.stock-card .name { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.stock-card .price { font-size: 1.2rem; font-weight: 700; color: white; }
.stock-card .change-up { color: #10b981; font-size: 0.85rem; font-weight: 600; }
.stock-card .change-down { color: #ef4444; font-size: 0.85rem; font-weight: 600; }

/* Recently Viewed Row */
.recent-row { display: flex; gap: 16px; overflow-x: auto; padding: 8px 0; }
.recent-item {
    text-align: center; min-width: 80px; padding: 8px 12px;
    background: #1e293b; border-radius: 10px; border: 1px solid #334155;
}
.recent-item .sym { font-size: 0.8rem; font-weight: 700; color: #e2e8f0; }
.recent-item .chg-up { font-size: 0.75rem; color: #10b981; font-weight: 600; }
.recent-item .chg-down { font-size: 0.75rem; color: #ef4444; font-weight: 600; }

/* Section Headers */
.section-head { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin: 1.2rem 0 0.6rem 0; }

/* Signal Cards */
.signal-buy { background: linear-gradient(135deg, #00b386, #10b981); color: white; padding: 1.2rem; border-radius: 14px; text-align: center; font-size: 1.2rem; font-weight: 700; box-shadow: 0 6px 24px rgba(0,179,134,0.3); }
.signal-sell { background: linear-gradient(135deg, #eb5b3c, #ef4444); color: white; padding: 1.2rem; border-radius: 14px; text-align: center; font-size: 1.2rem; font-weight: 700; box-shadow: 0 6px 24px rgba(235,91,60,0.3); }
.signal-hold { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; padding: 1.2rem; border-radius: 14px; text-align: center; font-size: 1.2rem; font-weight: 700; box-shadow: 0 6px 24px rgba(245,158,11,0.3); }

/* News Card */
.news-card { background: #1e293b; border-left: 4px solid #667eea; padding: 0.8rem 1rem; border-radius: 0 10px 10px 0; margin: 0.4rem 0; color: #e2e8f0; font-size: 0.9rem; }
.sentiment-pos { color: #10b981; font-weight: 700; }
.sentiment-neg { color: #ef4444; font-weight: 700; }
.sentiment-neu { color: #94a3b8; font-weight: 600; }

/* Index mini card */
.idx-card { background: #0f172a; border: 1px solid #1e293b; padding: 0.8rem; border-radius: 10px; margin: 0.3rem 0; }

/* Live Pulse Animations */
.pulse-green {
    background: #10b981; color: white; padding: 5px 15px; border-radius: 8px;
    font-weight: 800; font-size: 1.2rem; display: inline-block;
    animation: live-pulse-green 1.5s infinite;
    box-shadow: 0 0 15px rgba(16,185,129,0.5);
}
.pulse-red {
    background: #ef4444; color: white; padding: 5px 15px; border-radius: 8px;
    font-weight: 800; font-size: 1.2rem; display: inline-block;
    animation: live-pulse-red 1.5s infinite;
    box-shadow: 0 0 15px rgba(239,68,68,0.5);
}
@keyframes live-pulse-green {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16,185,129, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(16,185,129, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16,185,129, 0); }
}
@keyframes live-pulse-red {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239,68,68, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(239,68,68, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239,68,68, 0); }
}
</style>
""", unsafe_allow_html=True)


# ── Data Helpers ──────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_stock(raw_symbol, days=200, interval='1d', period=None):
    symbol = raw_symbol.strip().upper()
    mapped = None
    
    # Check exact match
    if symbol in STOCK_MAP:
        mapped = STOCK_MAP[symbol]
    else:
        # Check fuzzy match
        for k, v in STOCK_MAP.items():
            if k in symbol or symbol.replace(' ','') == k:
                mapped = v
                break
        
    if not mapped:
        mapped = symbol
        # Default to NSE if no suffix and not a known US stock
        us_stocks = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'UBER', 'SNAP', 'COIN', 'PLTR', 'DIS', 'BA', 'JPM', 'V', 'MA', 'KO', 'PEP', 'WMT', 'PG', 'JNJ'}
        if '.' not in mapped and '=' not in mapped and not mapped.startswith('^') and mapped not in us_stocks:
            mapped = mapped + '.NS'

    import time
    if not period: period = f'{days}d'
    
    for attempt in range(3):
        try:
            tk = yf.Ticker(mapped)
            df = tk.history(period=period, interval=interval)
            
            if df is None or df.empty:
                df = yf.download(mapped, period=period, interval=interval, progress=False)
                
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, mapped
        except Exception:
            pass
        time.sleep(1)
        
    return None, mapped

@st.cache_data(ttl=3600)
def fetch_fundamentals(mapped):
    """Fetch fundamental data as seen on Screener.in"""
    try:
        tk = yf.Ticker(mapped)
        info = tk.info
        return {
            'mkt_cap': info.get('marketCap', 0),
            'pe': info.get('trailingPE', 0),
            'pb': info.get('priceToBook', 0),
            'div_yield': info.get('dividendYield', 0),
            'high_52': info.get('fiftyTwoWeekHigh', 0),
            'low_52': info.get('fiftyTwoWeekLow', 0),
            'sector': info.get('sector', 'N/A'),
            'employees': info.get('fullTimeEmployees', 'N/A')
        }
    except:
        return None


@st.cache_data(ttl=300)
def get_usd_inr():
    try:
        fx = yf.download('USDINR=X', period='5d', progress=False)
        if fx is not None and not fx.empty:
            c = fx['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            return float(c.iloc[-1])
    except Exception:
        pass
    return 83.5


import re

@st.cache_data(ttl=5)
def get_realtime_price(symbol, mapped):
    """Fetch exact live price from Google Finance for near-zero delay"""
    # Map for Google Finance (e.g. NSE:RELIANCE)
    if mapped.endswith('.NS'): gf_sym = mapped.replace('.NS', '') + ':NSE'
    elif mapped.endswith('.BO'): gf_sym = mapped.replace('.BO', '') + ':BOM'
    elif mapped == '^NSEI': gf_sym = 'NIFTY_50:INDEXNSE'
    elif mapped == '^BSESN': gf_sym = 'SENSEX:INDEXBOM'
    elif mapped == '^NSEBANK': gf_sym = 'NIFTY_BANK:INDEXNSE'
    elif mapped in {'GC=F', 'SI=F', 'CL=F'}: return None # Skip commodities for now
    else: gf_sym = f"{mapped}:NASDAQ" # Default to NASDAQ for US stocks
    
    try:
        r = requests.get(f'https://www.google.com/finance/quote/{gf_sym}', timeout=3)
        if r.status_code == 200:
            m = re.search(r'data-last-price="([0-9.]+)"', r.text)
            if m: return float(m.group(1))
    except Exception:
        pass
    return None

def get_price_info(symbol, days=5):
    """Get current price, change, change% for a symbol."""
    df, mapped = fetch_stock(symbol, days)
    if df is None or df.empty:
        return None
    close = df['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) == 0:
        return None
    cur = float(close.iloc[-1])
    # Improved 'prev' logic: Try to get actual previous close from bar data
    prev = float(close.iloc[-2]) if len(close) >= 2 else cur
    
    # Try getting exact real-time price from Google Finance to fix Yahoo latency
    live_price = get_realtime_price(symbol, mapped)
    if live_price is not None:
        cur = live_price

    vol = df['Volume']
    if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
    last_vol = int(vol.iloc[-1]) if vol is not None and len(vol) > 0 else 0

    is_commodity = mapped in COMMODITY_USD
    is_indian = mapped.endswith('.NS') or mapped.endswith('.BO') or mapped.startswith('^')
    # If not US stock, default to ₹ for safety in Indian app context
    us_stocks = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'UBER', 'SNAP', 'COIN', 'PLTR', 'DIS', 'BA', 'JPM', 'V', 'MA', 'KO', 'PEP', 'WMT', 'PG', 'JNJ'}
    is_us = any(mapped.startswith(u) for u in us_stocks)

    if is_commodity:
        fx = get_usd_inr()
        cur *= fx
        prev *= fx
        curr_sym = '₹'
    elif is_indian or not is_us:
        curr_sym = '₹'
    else:
        curr_sym = '$'

    chg = cur - prev
    pct = (chg / prev * 100) if prev != 0 else 0
    return {'symbol': symbol, 'mapped': mapped, 'price': cur, 'prev': prev,
            'change': chg, 'pct': pct, 'currency': curr_sym, 'volume': last_vol}


def calculate_ut_bot(df, sensitivity=2, period=10):
    """
    Implements the UT Bot Alerts logic (ATR-based trailing stop).
    Returns the dataframe with 'UT_Trail', 'UT_Buy', and 'UT_Sell' columns.
    """
    if df is None or len(df) < period + 5:
        return df
    
    # Calculate ATR
    temp_df = df.copy()
    high = temp_df['High']
    low = temp_df['Low']
    close = temp_df['Close']
    
    # TR calculation
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    n_loss = sensitivity * atr
    
    trail = np.zeros(len(temp_df))
    src = close.values
    
    # Initialize first trail value to avoid zeros
    trail[0] = src[0]
    
    for i in range(1, len(temp_df)):
        prev_trail = trail[i-1]
        if src[i] > prev_trail and src[i-1] > prev_trail:
            trail[i] = max(prev_trail, src[i] - n_loss.iloc[i])
        elif src[i] < prev_trail and src[i-1] < prev_trail:
            trail[i] = min(prev_trail, src[i] + n_loss.iloc[i])
        elif src[i] > prev_trail:
            trail[i] = src[i] - n_loss.iloc[i]
        else:
            trail[i] = src[i] + n_loss.iloc[i]
            
    df = df.copy()
    df['UT_Trail'] = trail
    
    # Generate signals (Confirmed on bar close)
    df['UT_Buy'] = (df['Close'].shift(1) < df['UT_Trail'].shift(1)) & (df['Close'] > df['UT_Trail'])
    df['UT_Sell'] = (df['Close'].shift(1) > df['UT_Trail'].shift(1)) & (df['Close'] < df['UT_Trail'])
    
    return df

def detect_candle_pattern(df):
    """Detects confirmed candlestick patterns from the last CLOSED candle to avoid live repainting"""
    if df is None or len(df) < 3: return {"pattern": "Not enough data", "advice": "Wait for more data."}
    
    # We analyze the LAST CLOSED candle (-2) rather than the live fluctuating candle (-1)
    c = df.iloc[-2] 
    p = df.iloc[-3]
    
    # Current and previous fully closed OHLC
    cO, cH, cL, cC = float(c['Open']), float(c['High']), float(c['Low']), float(c['Close'])
    pO, pH, pL, pC = float(p['Open']), float(p['High']), float(p['Low']), float(p['Close'])
    
    cBody = abs(cC - cO)
    pBody = abs(pC - pO)
    cRange = cH - cL if (cH - cL) > 0 else 0.0001
    
    # Doji
    if cBody / cRange < 0.1:
        return {"pattern": "Doji ⚖️ (Indecision)", "advice": "Hold. Wait for breakout above Doji high or breakdown below low."}
        
    # Bullish Engulfing
    if pC < pO and cC > cO and cO <= pC and cC >= pO:
        return {"pattern": "Bullish Engulfing 📈", "advice": f"**BUY ENTRY**: Buy if next candle crosses {cH:,.2f}. **STOP LOSS**: {cL:,.2f}"}
        
    # Bearish Engulfing
    if pC > pO and cC < cO and cO >= pC and cC <= pO:
        return {"pattern": "Bearish Engulfing 📉", "advice": f"**SELL ENTRY**: Short if next candle goes below {cL:,.2f}. **STOP LOSS**: {cH:,.2f}"}
        
    # Shadows
    lower_shadow = min(cO, cC) - cL
    upper_shadow = cH - max(cO, cC)
    
    # Hammer
    if lower_shadow > 2.0 * cBody and upper_shadow < cBody * 0.5:
        if cC > cO: return {"pattern": "Bullish Hammer 🔨", "advice": f"**BUY ENTRY**: Buy if next candle crosses {cH:,.2f}. **STOP LOSS**: {cL:,.2f}"}
        else: return {"pattern": "Hanging Man 🔻", "advice": f"**SELL ENTRY**: Wait for next candle to close below {cL:,.2f} before shorting."}
        
    # Shooting Star / Inverted Hammer
    if upper_shadow > 2.5 * cBody and lower_shadow < cBody * 0.5:
        if cC < cO: return {"pattern": "Shooting Star 🌠", "advice": f"**SELL ENTRY**: Short if next candle goes below {cL:,.2f}. **STOP LOSS**: {cH:,.2f}"}
        else: return {"pattern": "Inverted Hammer ⛏️", "advice": f"**BUY ENTRY**: Buy if next candle crosses {cH:,.2f}. **STOP LOSS**: {cL:,.2f}"}
        
    # Marubozu (Strong momentum)
    if cBody / cRange > 0.9:
        if cC > cO: return {"pattern": "Bullish Marubozu 🧨", "advice": "STRONG BUY: High momentum upward. Trail stop loss deeply."}
        else: return {"pattern": "Bearish Marubozu 🧱", "advice": "STRONG SELL: Heavy selling pressure. Avoid buying."}
        
    if cC > cO: return {"pattern": "Standard Bullish 🟢", "advice": "Trend is UP. Consider buying on minor intraday dips."}
    elif cC < cO: return {"pattern": "Standard Bearish 🔴", "advice": "Trend is DOWN. Avoid fresh buying."}
    
    return {"pattern": "Neutral ➖", "advice": "No clear breakout pattern right now."}

# ── News Fetchers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_market_news(query="Indian Stock Market"):
    """Fetches highly robust targeted news using Google News RSS"""
    try:
        q = urllib.parse.quote_plus(query)
        url = f'https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        u = urllib.request.urlopen(req, timeout=10)
        tree = ET.parse(u)
        items = []
        for item in tree.findall('.//item')[:15]:
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            title = title.split(' - ')[0] if ' - ' in title else title # Clean source suffix
            if len(title) > 10:
                items.append({'title': title, 'url': link, 'source': 'GoogleNews'})
        return items
    except Exception:
        return []

# ── Sentiment ─────────────────────────────────────────────────────────────
POS_WORDS = ['rally','gain','surge','bullish','record','high','jump','soar','beat',
             'outperform','buy','upgrade','profit','growth','boom','recover','strong','rise','up']
NEG_WORDS = ['fall','drop','crash','bearish','low','plunge','miss','sell','downgrade',
             'loss','decline','weak','cut','fear','risk','slump','down','tank','tumble']

def score_headline(text):
    t = text.lower()
    return sum(1 for w in POS_WORDS if w in t) - sum(1 for w in NEG_WORDS if w in t)

def analyze_live_candle(df):
    """Analyzes the current forming candle for immediate momentum"""
    if df is None or len(df) < 2: return {"bias": "Neutral", "color": "#94a3b8", "desc": "Stabilizing...", "pct": 0}
    
    live = df.iloc[-1]
    prev = df.iloc[-2]
    
    lO, lH, lL, lC = float(live['Open']), float(live['High']), float(live['Low']), float(live['Close'])
    pH, pL = float(prev['High']), float(prev['Low'])
    
    pct_from_open = (lC - lO) / lO * 100 if lO != 0 else 0
    
    if lC > lO:
        bias = "UP 📈"
        color = "#10b981"
        if lC > pH: desc = "Strong Bullish Breakout"
        else: desc = "Bullish Accumulation"
    elif lC < lO:
        bias = "DOWN 📉"
        color = "#ef4444"
        if lC < pL: desc = "Strong Bearish Breakdown"
        else: desc = "Bearish Pressure"
    else:
        bias = "NEUTRAL ⚖️"
        color = "#94a3b8"
        desc = "Price Indecision"
        
    return {"bias": bias, "color": color, "desc": desc, "pct": pct_from_open}

def analyze_news(headlines):
    if not headlines: return 0.0, []
    scored = []
    for h in headlines:
        title = h.get('title', '') if isinstance(h, dict) else str(h)
        sc = score_headline(title)
        label = 'positive' if sc > 0 else 'negative' if sc < 0 else 'neutral'
        entry = {**(h if isinstance(h, dict) else {'title': title}), 'score': sc, 'label': label}
        scored.append(entry)
    avg = sum(s['score'] for s in scored) / len(scored)
    return round(avg, 3), scored


# ── TradingView Data Source ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_tv_sentiment(symbol, mapped):
    """
    Fetches the Technical Analysis summary from TradingView.
    """
    tv_sym = get_tv_symbol(symbol, mapped)
    try:
        from tradingview_ta import TA_Handler, Interval
        handler = TA_Handler(
            symbol=tv_sym.split(':')[-1],
            exchange=tv_sym.split(':')[0],
            screener="india" if "NSE" in tv_sym or "BSE" in tv_sym else "america",
            interval=Interval.INTERVAL_1_DAY
        )
        analysis = handler.get_analysis()
        summary = analysis.summary
        buy = summary.get('BUY', 0)
        sell = summary.get('SELL', 0)
        total = sum(summary.values())
        return (buy - sell) / total if total > 0 else 0.0
    except Exception as e:
        # Fallback to 0 if library is missing or symbol is not found
        return 0.0


# ── AI Engine ─────────────────────────────────────────────────────────────
class AIEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    @staticmethod
    def _rsi(prices, period=14):
        if len(prices) < 2: return 50
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        ag = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains) if len(gains) > 0 else 0
        al = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses) if len(losses) > 0 else 0
        if al == 0: return 100
        return 100 - (100 / (1 + ag / al))

    def _features(self, prices, volumes, idx, lb=20):
        w = prices[max(0, idx-lb):idx]
        vw = volumes[max(0, idx-lb):idx]
        if len(w) < 5: return None
        ma5 = np.mean(w[-5:]); ma10 = np.mean(w[-10:]) if len(w)>=10 else np.mean(w); ma20 = np.mean(w)
        std5 = np.std(w[-5:]); std20 = np.std(w)
        mom = (w[-1]-w[0])/w[0] if w[0]!=0 else 0
        rsi = self._rsi(w)
        macd = ma5 - ma20
        va = np.mean(vw) if len(vw)>0 else 0
        vc = (vw[-1]-vw[0])/vw[0] if len(vw)>0 and vw[0]!=0 else 0
        bbu = ma20+2*std20; bbl = ma20-2*std20
        bbp = (w[-1]-bbl)/(bbu-bbl) if bbu!=bbl else 0.5
        pmr = w[-1]/ma20 if ma20!=0 else 1
        return [ma5,ma10,ma20,std5,std20,mom,rsi,macd,va,vc,bbp,pmr]

    def train(self, symbol, prices, volumes, news_sent=0.0):
        prices = np.array(prices, dtype=float); volumes = np.array(volumes, dtype=float)
        X, y1, y2, y3, y4 = [], [], [], [], []
        for i in range(25, len(prices)-4):
            f = self._features(prices, volumes, i)
            if f is None: continue
            X.append(f)
            y1.append(1 if prices[i+1]>prices[i] else 0) # 1 step
            y2.append(1 if prices[i+2]>prices[i] else 0) # 2 steps
            y3.append(1 if prices[i+3]>prices[i] else 0) # 3 steps
            y4.append(1 if prices[i+4]>prices[i] else 0) # 4 steps
            
        if len(X) < 30: return None
        X = np.array(X)
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        
        self.models[symbol] = {'d1': {}, 'd2': {}, 'd3': {}, 'd4': {}}
        for day, labels in zip(['d1','d2','d3','d4'], [y1, y2, y3, y4]):
            y = np.array(labels)
            Xtr,Xte,ytr,yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
            rf.fit(Xtr,ytr); gb.fit(Xtr,ytr)
            self.models[symbol][day] = {'rf': rf, 'gb': gb, 'acc': (rf.score(Xte,yte) + gb.score(Xte,yte))/2}
            
        self.scalers[symbol] = sc
        return {'d1_acc': self.models[symbol]['d1']['acc'], 'd2_acc': self.models[symbol]['d2']['acc'], 'd3_acc': self.models[symbol]['d3']['acc'], 'd4_acc': self.models[symbol]['d4']['acc']}

    def predict(self, symbol, prices, volumes, news_sent=0.0, tv_sent=0.0, intraday=False):
        if symbol not in self.models: return None
        prices = np.array(prices, dtype=float); volumes = np.array(volumes, dtype=float)
        
        f_latest = self._features(prices, volumes, len(prices))
        if f_latest is None: return None
        Xs_latest = self.scalers[symbol].transform([f_latest])
        
        # We only need f_prev if we're doing the 'Daily' Tomorrow/Day After logic
        Xs_prev = None
        if not intraday:
            f_prev = self._features(prices, volumes, len(prices)-1)
            if f_prev is not None:
                Xs_prev = self.scalers[symbol].transform([f_prev])
            else:
                Xs_prev = Xs_latest # Fallback if prev is missing
        
        steps = ['d1', 'd2', 'd4'] if intraday else ['d1', 'd1', 'd2']
        feats = [Xs_latest, Xs_latest, Xs_latest] if intraday else [Xs_prev, Xs_latest, Xs_latest]
        labels = ['today', 'tomorrow', 'day_after']
        
        results = {}
        for label, step_key, feat in zip(labels, steps, feats):
            m_set = self.models[symbol][step_key]
            probs = [m_set[m].predict_proba(feat)[0][1] for m in ['rf', 'gb']]
            # Incorporate weighted sentiment from News (8%) and TradingView (12%)
            up_prob = np.clip(np.mean(probs) + 0.08*news_sent + 0.12*tv_sent, 0, 1)
            dn_prob = 1 - up_prob
            sig = 'BUY' if up_prob > 0.55 else 'SELL' if dn_prob > 0.55 else 'HOLD'
            results[label] = {
                'signal': sig, 
                'confidence': round(max(up_prob, dn_prob), 4), 
                'up_prob': round(up_prob, 4),
                'news_bias': round(0.08*news_sent, 4),
                'tv_bias': round(0.12*tv_sent, 4)
            }
        return results


# ── Charts ────────────────────────────────────────────────────────────────
def build_candle_chart(df, symbol):
    # Groww Theme Colors
    UP_COLOR = '#00b386'
    DOWN_COLOR = '#eb5b3c'
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
        increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        name='Price'
    ), row=1, col=1)
    
    # Simple Moving Average (subtle)
    ma20 = df['Close'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color='#667eea', width=1, dash='dot'), name='MA20', opacity=0.6), row=1, col=1)
    
    # --- UT Bot Alerts Visualization ---
    if 'UT_Trail' in df.columns:
        # Trailing Stop Line
        fig.add_trace(go.Scatter(x=df.index, y=df['UT_Trail'], 
                                 line=dict(color='#94a3b8', width=1, dash='dot'), 
                                 name='UT Trail', opacity=0.4), row=1, col=1)
        
        # Buy Signals (Triangles below low)
        buys = df[df['UT_Buy'] == True]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys.index, y=buys['Low'] * 0.995, 
                                     mode='markers',
                                     marker=dict(symbol='triangle-up', size=12, color='#00b386', 
                                                 line=dict(width=1, color='white')),
                                     name='UT Buy'), row=1, col=1)
                                     
        # Sell Signals (Triangles above high)
        sells = df[df['UT_Sell'] == True]
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells.index, y=sells['High'] * 1.005, 
                                     mode='markers',
                                     marker=dict(symbol='triangle-down', size=12, color='#eb5b3c', 
                                                 line=dict(width=1, color='white')),
                                     name='UT Sell'), row=1, col=1)
    
    # Volume
    colors = [UP_COLOR if c >= o else DOWN_COLOR for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Vol', opacity=0.3), row=2, col=1)
    
    fig.update_layout(
        template='plotly_white', height=450, showlegend=False,
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        margin=dict(l=5, r=5, t=35, b=5)
    )
    
    # Clean up axes (Groww style minimalism)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=False, showline=False, side='right')
    
    return fig

def build_gauge(up_prob, signal, title="Signal"):
    cmap = {'BUY':'#10b981','SELL':'#ef4444','HOLD':'#f59e0b'}
    fig = go.Figure(go.Indicator(mode="gauge+number", value=up_prob*100,
        title={'text':f"{title}: {signal}",'font':{'size':16,'color':'white'}},
        number={'suffix':'%','font':{'color':'white'}},
        gauge={'axis':{'range':[0,100],'tickcolor':'white'}, 'bar':{'color':cmap.get(signal,'#f59e0b')},
               'bgcolor':'#1e293b',
               'steps':[{'range':[0,45],'color':'rgba(239,68,68,0.15)'},
                        {'range':[45,55],'color':'rgba(245,158,11,0.15)'},
                        {'range':[55,100],'color':'rgba(16,185,129,0.15)'}]}))
    fig.update_layout(height=250, paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                      font={'color':'white'}, margin=dict(l=30,r=30,t=50,b=10))
    return fig

def get_tv_symbol(symbol, mapped):
    """Map yfinance symbols to TradingView symbols"""
    if mapped.endswith('.NS'): return "NSE:" + mapped.replace('.NS', '')
    if mapped.endswith('.BO'): return "BSE:" + mapped.replace('.BO', '')
    if mapped == '^NSEI': return "NSE:NIFTY"
    if mapped == '^BSESN': return "BSE:SENSEX"
    if mapped == '^NSEBANK': return "NSE:BANKNIFTY"
    if mapped == 'GC=F': return "COMEX:GC1!"
    if mapped == 'SI=F': return "COMEX:SI1!"
    if mapped == 'CL=F': return "NYMEX:CL1!"
    return mapped

def build_tradingview_chart(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container" style="height:800px;width:100%">
      <div id="tradingview_{symbol.lower()}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": 800,
        "symbol": "{tv_sym}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_{symbol.lower()}"
      }});
      </script>
    </div>
    """

def build_tradingview_analysis(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
        "interval": "1D",
        "width": "100%",
        "isTransparent": false,
        "height": 450,
        "symbol": "{tv_sym}",
        "showIntervalTabs": true,
        "displayMode": "single",
        "locale": "en",
        "colorTheme": "dark"
      }}
      </script>
    </div>
    """

def build_tradingview_profile_widget(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
      {{
        "width": "100%",
        "height": 400,
        "colorTheme": "dark",
        "isTransparent": false,
        "symbol": "{tv_sym}",
        "locale": "en"
      }}
      </script>
    </div>
    """

def build_tradingview_news_widget(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {{
        "feedMode": "symbol",
        "symbol": "{tv_sym}",
        "isTransparent": false,
        "displayMode": "regular",
        "width": "100%",
        "height": 450,
        "colorTheme": "dark",
        "locale": "en"
      }}
      </script>
    </div>
    """

def build_tradingview_market_news():
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {{
        "feedMode": "market",
        "market": "stock",
        "isTransparent": false,
        "displayMode": "regular",
        "width": "100%",
        "height": 600,
        "colorTheme": "dark",
        "locale": "en"
      }}
      </script>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
def main():
    st.markdown('<div class="main-title">🧠 AI Market Predictor Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time BSE • NSE • MoneyControl — AI-Powered Predictions &nbsp;'
                '<span class="live-badge">● LIVE</span></div>', unsafe_allow_html=True)

    if 'engine' not in st.session_state:
        st.session_state.engine = AIEngine()

    # ── Ticker Bar & Live Refresh ─────────────────────────────────────
    ticker_syms = ['NIFTY', 'SENSEX', 'BANKNIFTY']
    lc1, lc2 = st.columns([5, 1])
    with lc2:
        live_mode = st.checkbox("📡 Live Mode", help="Auto-refreshes every 30s")
    
    if live_mode:
        st.markdown("""
            <script>
            setTimeout(function(){
               window.location.reload();
            }, 30000);
            </script>
        """, unsafe_allow_html=True)

    ticker_html = '<div class="ticker-bar">'
    for ts in ticker_syms:
        info = get_price_info(ts, 5)
        if info:
            cls = 'ticker-up' if info['change'] >= 0 else 'ticker-down'
            arrow = '▲' if info['change'] >= 0 else '▼'
            ticker_html += (f'<span class="ticker-item"><span class="ticker-name">{ts}</span>'
                           f'<span class="ticker-price">{info["currency"]}{info["price"]:,.2f}</span>'
                           f'<span class="{cls}">{arrow} {info["change"]:+,.2f} ({info["pct"]:+.2f}%)</span></span>')
    ticker_html += '</div>'
    st.markdown(ticker_html, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🧠 AI Predictor Pro")
        page = st.radio("Navigate", [
            "🏠 Explore", "🔮 AI Prediction", "🔍 Stock Screener", "📰 Market News",
            "📊 All Stocks", "🏆 Top Movers", "📈 Sector View"
        ])
        st.markdown("---")
        st.caption("**Data Sources**")
        st.caption("✅ Yahoo Finance Live")
        st.caption("✅ MoneyControl News")
        st.caption("✅ TradingView Insights")
        st.caption("✅ Screener.in Fundamentals")
        st.caption("✅ BSE / NSE India")

    # ── Pages ─────────────────────────────────────────────────────────
    if page == "🏠 Explore":
        page_explore()
    elif page == "🔮 AI Prediction":
        page_prediction()
    elif page == "🔍 Stock Screener":
        page_screener()
    elif page == "📰 Market News":
        page_news()
    elif page == "📊 All Stocks":
        page_all_stocks()
    elif page == "🏆 Top Movers":
        page_top_movers()
    elif page == "📈 Sector View":
        page_sector_view()

    st.markdown("---")
    st.caption("🧠 AI Market Predictor Pro • BSE • NSE • MoneyControl")

def build_tradingview_news_widget(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {{
        "feedMode": "symbol",
        "symbol": "{tv_sym}",
        "isTransparent": false,
        "displayMode": "regular",
        "width": "100%",
        "height": 450,
        "colorTheme": "dark",
        "locale": "en"
      }}
      </script>
    </div>
    """

def build_tradingview_market_news():
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {{
        "feedMode": "market",
        "market": "stock",
        "isTransparent": false,
        "displayMode": "regular",
        "width": "100%",
        "height": 600,
        "colorTheme": "dark",
        "locale": "en"
      }}
      </script>
    </div>
    """


# ── PAGE: Explore (Groww-style) ───────────────────────────────────────────
def page_explore():
    # Watchlist / Recently Viewed
    st.markdown('<div class="section-head">📌 Watchlist</div>', unsafe_allow_html=True)
    row_html = '<div class="recent-row">'
    for sym in WATCHLIST_DEFAULT:
        info = get_price_info(sym, 5)
        if info:
            cls = 'chg-up' if info['pct'] >= 0 else 'chg-down'
            row_html += (f'<div class="recent-item"><div class="sym">{sym}</div>'
                        f'<div class="{cls}">{info["pct"]:+.2f}%</div></div>')
    row_html += '</div>'
    st.markdown(row_html, unsafe_allow_html=True)

    # NEW: Market Pulse
    st.markdown('<div class="section-head">⚡ Market Pulse (Candlestick & UT Bot Alert)</div>', unsafe_allow_html=True)
    pulse_syms = ['RELIANCE', 'TATAMOTORS', 'WIPRO', 'GOLD']
    pulse_cols = st.columns(len(pulse_syms))
    for i, psym in enumerate(pulse_syms):
        pdf, _ = fetch_stock(psym, 30)
        if pdf is not None:
            pat = detect_candle_pattern(pdf)
            pdf = calculate_ut_bot(pdf)
            ut_s = "BUY 🚀" if pdf.iloc[-1]['Close'] > pdf.iloc[-1]['UT_Trail'] else "SELL 💥"
            color = "#10b981" if "BUY" in ut_s else "#ef4444"
            with pulse_cols[i]:
                st.markdown(f"""<div class="stock-card" style="border-left: 4px solid {color}">
                    <div class="name">{psym}</div>
                    <div style="font-size:0.8rem; color:{color}; font-weight:700">{pat['pattern']}</div>
                    <div style="font-size:0.75rem; color:{color}; opacity:0.8">UT Bot: {ut_s}</div>
                </div>""", unsafe_allow_html=True)

    # Most Traded (top 4 cards)
    st.markdown('<div class="section-head">🔥 Most Traded Stocks</div>', unsafe_allow_html=True)
    top4 = ['RELIANCE', 'TATASTEEL', 'SBIN', 'ADANIGREEN']
    cols = st.columns(4)
    for i, sym in enumerate(top4):
        info = get_price_info(sym, 5)
        if info:
            cls = 'change-up' if info['change'] >= 0 else 'change-down'
            with cols[i]:
                st.markdown(f"""<div class="stock-card">
                    <div class="name">{sym}</div>
                    <div class="price">{info['currency']}{info['price']:,.2f}</div>
                    <div class="{cls}">{info['change']:+.2f} ({info['pct']:+.2f}%)</div>
                </div>""", unsafe_allow_html=True)

    # Top Movers Table (Gainers)
    st.markdown('<div class="section-head">📈 Top Movers Today</div>', unsafe_allow_html=True)
    movers_syms = ['TATAMOTORS','TATASTEEL','TATAPOWER','ADANIGREEN','ADANIENT',
                   'RELIANCE','SBIN','ICICIBANK','INFY','WIPRO','BAJFINANCE','HDFCBANK',
                   'ITC','MARUTI','JSWSTEEL','VEDL','NIPPON','COALINDIA']
    movers_data = []
    prog = st.progress(0, text="Fetching top movers...")
    for idx, sym in enumerate(movers_syms):
        prog.progress((idx+1)/len(movers_syms), text=f"Fetching {sym}...")
        info = get_price_info(sym, 5)
        if info:
            movers_data.append(info)
    prog.empty()

    if movers_data:
        tab1, tab2, tab3 = st.tabs(["🟢 Gainers", "🔴 Losers", "📊 All"])
        gainers = sorted([m for m in movers_data if m['pct'] > 0], key=lambda x: -x['pct'])
        losers = sorted([m for m in movers_data if m['pct'] < 0], key=lambda x: x['pct'])

        with tab1:
            if gainers:
                gdf = pd.DataFrame([{'Company': g['symbol'], f'Price': f"{g['currency']}{g['price']:,.2f}",
                    'Change': f"{g['change']:+.2f}", 'Change%': f"{g['pct']:+.2f}%",
                    'Volume': f"{g['volume']:,}"} for g in gainers])
                st.dataframe(gdf, use_container_width=True, hide_index=True)
            else:
                st.info("No gainers found")

        with tab2:
            if losers:
                ldf = pd.DataFrame([{'Company': l['symbol'], f'Price': f"{l['currency']}{l['price']:,.2f}",
                    'Change': f"{l['change']:+.2f}", 'Change%': f"{l['pct']:+.2f}%",
                    'Volume': f"{l['volume']:,}"} for l in losers])
                st.dataframe(ldf, use_container_width=True, hide_index=True)
            else:
                st.info("No losers found")

        with tab3:
            adf = pd.DataFrame([{'Company': m['symbol'], f'Price': f"{m['currency']}{m['price']:,.2f}",
                'Change': f"{m['change']:+.2f}", 'Change%': f"{m['pct']:+.2f}%"} for m in movers_data])
            st.dataframe(adf, use_container_width=True, hide_index=True)

    # Quick Links
    st.markdown('<div class="section-head">🔗 Products & Tools</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.link_button("📊 MoneyControl", "https://www.moneycontrol.com/", use_container_width=True)
    with lc2:
        st.link_button("📈 NSE India", "https://www.nseindia.com/", use_container_width=True)
    with lc3:
        st.link_button("🔍 Screener.in", "https://www.screener.in/", use_container_width=True)


# ── PAGE: AI Prediction ──────────────────────────────────────────────────
def page_prediction():
    st.subheader("🔮 AI Stock Prediction Engine (3-Day Forecast)")
    st.caption("This model analyzes historical trends, technical indicators, and news sentiment to predict if the price will go UP or DOWN for the next **3 days**.")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        symbol = st.text_input("Enter Stock Symbol",
            placeholder="RELIANCE, TATAMOTORS, GOLD, SILVER, NIPPON, AAPL...")
    with c2:
        st.write(""); st.write("")
        run = st.button("🧠 Run AI", use_container_width=True)
    with c3:
        st.write(""); st.write("")
        refresh = st.button("🔄 Refresh", use_container_width=True)
    
    if refresh:
        st.cache_data.clear()
        st.success("♻️ Cache Cleared. Fetching fresh data...")

    # Show available symbols
    with st.expander("📋 Available Symbols"):
        sym_list = sorted(STOCK_MAP.keys())
        st.write(", ".join(sym_list))

    if run and symbol.strip():
        symbol = symbol.strip().upper()
        with st.spinner(f"📡 Fetching data for {symbol}..."):
            df, mapped = fetch_stock(symbol, 250)
        
        # Try getting exact real-time price from Google Finance first
        live_price = get_realtime_price(symbol, mapped)
        
        if (df is None or df.empty) and live_price is None:
            st.error(f"❌ No data for {symbol}. Check the symbol name.")
            return

        is_commodity = mapped in COMMODITY_USD
        is_indian = mapped.endswith('.NS') or mapped.endswith('.BO') or mapped.startswith('^')
        us_stocks = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'UBER', 'SNAP', 'COIN', 'PLTR', 'DIS', 'BA', 'JPM', 'V', 'MA', 'KO', 'PEP', 'WMT', 'PG', 'JNJ'}
        is_us = any(mapped.startswith(u) for u in us_stocks)
        
        curr = '₹' if (is_indian or not is_us) else '$'
        
        # Display Current Price Card
        if live_price is not None:
            cur = live_price
            prev = live_price  # fallback
            if df is not None and not df.empty:
                close = df['Close']
                if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
                prev = close.dropna().iloc[-1] if len(close) > 0 else live_price
            
            chg = cur - prev
            pct = (chg / prev * 100) if prev != 0 else 0
            cls = 'change-up' if chg >= 0 else 'change-down'
            
            ut_s = "BUY 🚀" if cur > df_run.iloc[-1]['UT_Trail'] else "SELL 💥"
            ut_color = "#10b981" if "BUY" in ut_s else "#ef4444"
            
            st.markdown(f"""<div class="stock-card" style="padding:1.5rem; border-right: 10px solid {ut_color}">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <div>
                        <div class="name" style="font-size:1rem">📊 {symbol}</div>
                        <div class="price" style="font-size:2rem">{curr}{cur:,.2f}</div>
                        <div class="{cls}" style="font-size:1rem">{'▲' if chg>=0 else '▼'} {chg:+,.2f} ({pct:+.2f}%)</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase">UT Bot Indicator</div>
                        <div style="font-size:1.5rem; color:{ut_color}; font-weight:900">{ut_s}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            # NEW: Screener.in Deep Fundamentals Button
            clean_sym = mapped.replace('.NS', '').replace('.BO', '')
            st.link_button(f"🌐 Visit Full Screener.in Profile for {symbol}", 
                           f"https://www.screener.in/company/{clean_sym}/", 
                           use_container_width=True)

            # NEW: Fundamental Pulse (Screener.in Data Source)
            with st.spinner("📊 Fetching Screener fundamentals..."):
                f_data = fetch_fundamentals(mapped)
            
            if f_data:
                st.markdown('<div class="section-head">🏛️ Screener.in Fundamentals</div>', unsafe_allow_html=True)
                fc1, fc2, fc3, fc4 = st.columns(4)
                
                # Format Market Cap to Cr (Indian style)
                mcap_in_cr = f_data['mkt_cap'] / 10000000 if "₹" in curr else f_data['mkt_cap']
                mcap_unit = "Cr" if "₹" in curr else ""
                
                fc1.metric("Market Cap", f"{mcap_in_cr:,.0f} {mcap_unit}")
                fc2.metric("P/E Ratio", f"{f_data['pe']:.2f}" if f_data['pe'] else "N/A")
                fc3.metric("Div. Yield", f"{f_data['div_yield']*100:.2f}%" if f_data['div_yield'] else "0.0%")
                fc4.metric("P/B Ratio", f"{f_data['pb']:.2f}" if f_data['pb'] else "N/A")
                
                st.markdown(f"**Sector:** {f_data['sector']} | **52W High:** {curr}{f_data['high_52']:,.2f} | **52W Low:** {curr}{f_data['low_52']:,.2f}")
                st.write("") # Spacer

            # NEW: AI Prediction Mode
            pred_mode = st.radio("Prediction Mode:", ["📅 Daily (Swing Trade)", "📈 Intraday (Next 1 Hour)"], horizontal=True)
            
            # Re-fetch for Intraday mode
            if "Intraday" in pred_mode:
                with st.spinner("⚡ Fetching 15m intraday data..."):
                    df_run, _ = fetch_stock(symbol, interval='15m', period='5d') # Reduced from 7d to 5d for faster mobile loading
            else:
                df_run = df

            # AI Prediction Logic
            if df_run is not None and len(df_run) > 30:
                # --- UT Bot Alerts Integration ---
                df_run = calculate_ut_bot(df_run)
                
                close = df_run['Close']; vol = df_run['Volume']
                if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
                if isinstance(vol, pd.DataFrame): vol = vol.iloc[:,0]
                prices = close.dropna().astype(float).tolist()
                volumes = vol.dropna().astype(float).tolist()

                if is_commodity:
                    fx = get_usd_inr(); prices = [p*fx for p in prices]

                with st.spinner("📰 Analyzing news..."): 
                    news = fetch_market_news(f"{symbol} share stock market news")
                    sent, scored_news = analyze_news(news)

                with st.spinner("🧠 Training AI ensemble..."):
                    metrics = st.session_state.engine.train(symbol, prices, volumes, sent)
                
                if metrics:
                    with st.spinner("📡 Fetching TradingView Sentiment..."):
                        tv_sentiment = fetch_tv_sentiment(symbol, mapped)
                        
                    is_intra = "Intraday" in pred_mode
                    pred = st.session_state.engine.predict(symbol, prices, volumes, sent, tv_sent=tv_sentiment, intraday=is_intra)
                    if pred:
                        pass # Moved to bottom
                    else:
                        st.warning("⚠️ Prediction failed for this stock.")
                else:
                    st.warning("⚠️ Not enough historical data for AI prediction (need 60+ days). Showing live price and chart only.")

                # MOVED: TradingView Integration (now always visible if data exists)
                st.markdown('<div class="section-head">📺 TradingView Live Chart & Deep Analysis</div>', unsafe_allow_html=True)
                tv_tab1, tv_tab2, tv_tab3, tv_tab4, tv_tab5 = st.tabs(["📊 Advanced Chart", "🔍 Stock Profile", "📈 Technical Pulse", "📰 TradingView News", "🤖 AI vs TradingView"])
                with tv_tab1:
                    st.components.v1.html(build_tradingview_chart(symbol, mapped), height=850)
                with tv_tab2:
                    st.components.v1.html(build_tradingview_profile_widget(symbol, mapped), height=400)
                with tv_tab3:
                    st.components.v1.html(build_tradingview_analysis(symbol, mapped), height=450)
                with tv_tab4:
                    st.components.v1.html(build_tradingview_news_widget(symbol, mapped), height=450)
                with tv_tab5:
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("#### 📺 TradingView Technicals")
                        st.components.v1.html(build_tradingview_analysis(symbol, mapped), height=450)
                    with sc2:
                        st.markdown("#### 🤖 Antigravity AI Prediction")
                        if metrics and 'pred' in locals() and pred:
                            st.plotly_chart(build_gauge(pred['today']['up_prob'], pred['today']['signal'], "AI Signal"), use_container_width=True)
                            st.markdown(f"**AI Confidence:** {pred['today']['confidence']:.1%}")
                            st.markdown(f"**Market Sentiment:** {sent:+.2f}")
                        else:
                            st.info("AI Prediction not available for this stock yet (Training...)")

                st.write("")
                # Instruction for the user's pip error
                try:
                    from tradingview_ta import TA_Handler
                except ImportError:
                    st.warning("💡 **TradingView TA Library Missing**: Your system is having trouble installing dependencies automatically. Please run this command in your terminal: `python -m pip install tradingview-ta` to enable full mathematical integration.")

                st.write("")

                st.write("")
                # --- NEW: 3-Day Forecast Signals ---
                st.markdown('<div class="section-head">🎯 AI Signal Dashboard (3-Day Forecast)</div>', unsafe_allow_html=True)
                tc1, tc2, tc3 = st.columns(3)
                sig_cls = {'BUY': 'signal-buy', 'SELL': 'signal-sell', 'HOLD': 'signal-hold'}
                sig_emoji = {'BUY': '🚀 BUY', 'SELL': '💥 SELL', 'HOLD': '⚖️ HOLD'}
                
                with tc1:
                    st.markdown(f'<div class="{sig_cls[pred["today"]["signal"]]}">TODAY<br><span style="font-size:1.4rem;">{sig_emoji[pred["today"]["signal"]]}</span><br>'
                                f'<span style="font-size:0.8rem;opacity:0.9;">Confidence: {pred["today"]["confidence"]:.1%}</span></div>', unsafe_allow_html=True)
                with tc2:
                    st.markdown(f'<div class="{sig_cls[pred["tomorrow"]["signal"]]}">TOMORROW<br><span style="font-size:1.4rem;">{sig_emoji[pred["tomorrow"]["signal"]]}</span><br>'
                                f'<span style="font-size:0.8rem;opacity:0.9;">Confidence: {pred["tomorrow"]["confidence"]:.1%}</span></div>', unsafe_allow_html=True)
                with tc3:
                    st.markdown(f'<div class="{sig_cls[pred["day_after"]["signal"]]}">DAY 3<br><span style="font-size:1.4rem;">{sig_emoji[pred["day_after"]["signal"]]}</span><br>'
                                f'<span style="font-size:0.8rem;opacity:0.9;">Confidence: {pred["day_after"]["confidence"]:.1%}</span></div>', unsafe_allow_html=True)

                st.write("")
                # --- VERTICAL LAYOUT: Analysis Below Chart ---
                st.markdown("### 📊 Internal Plotly Chart")
                st.plotly_chart(build_candle_chart(df_run.tail(60), symbol), use_container_width=True)
                
                # Analysis Section below the chart
                st_live_col, st_conf_col = st.columns(2)
                
                # Confirmed Pattern (Closed)
                res = detect_candle_pattern(df_run.tail(3))
                # Live Pulse (Currently Forming)
                live_res = analyze_live_candle(df_run)
                p_cls = "pulse-green" if "UP" in live_res["bias"] else "pulse-red" if "DOWN" in live_res["bias"] else ""
                
                with st_live_col:
                    st.markdown(f'''
                    <div style="background:#1e293b; color:#e2e8f0; padding:15px; border-radius:10px; border:1px solid #334155; margin-bottom:15px; height:220px;">
                        <h4 style="margin-top:0; color:#94a3b8; font-size:0.8rem; text-transform:uppercase;">Live Candle Monitor</h4>
                        <div style="text-align:center; padding:10px 0;">
                            <div class="{p_cls}">{live_res["bias"]}</div>
                            <div style="font-size:1.5rem; font-weight:800; color:{live_res["color"]}; margin-top:8px;">{live_res["pct"]:+.2f}%</div>
                            <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">{live_res["desc"]}</div>
                        </div>
                    </div>''', unsafe_allow_html=True)

                with st_conf_col:
                    st.markdown(f'''
                    <div style="background:#1e293b; color:#e2e8f0; padding:15px; border-radius:10px; border:1px solid #334155; height:220px;">
                        <h4 style="margin-top:0; color:#94a3b8; font-size:0.8rem; text-transform:uppercase;">Confirmed Analysis</h4>
                        <b>Recent Pattern:</b> {res["pattern"]} <br>
                        <b>Suggested Strategy:</b> <br><span style="font-size:0.9rem">{res["advice"]}</span>
                    </div>''', unsafe_allow_html=True)

                st.markdown(f'''
                <div style="background:#1e293b; color:#e2e8f0; padding:15px; border-radius:10px; border:1px solid #334155; width:100%;">
                    <h4 style="margin-top:0; color:#94a3b8; font-size:0.8rem; text-transform:uppercase;">Fundamental & Technical Deep Insight</h4>
                    <span style="font-size:0.85rem; color:#94a3b8">
                    Our engine detected:
                    <ul style="padding-left:15px; margin:5px 0;">
                        <li>Technical Rating: <b>{tv_sentiment:+.2f}</b></li>
                        <li>News Sentiment: <b>{sent:+.2f}</b></li>
                        <li>UT Bot Signal: <b>{"BUY 🚀" if df_run.iloc[-1]["Close"] > df_run.iloc[-1]["UT_Trail"] else "SELL 💥"}</b></li>
                        <li>AI Priority: <b style="color:#667eea">{pred["today"]["signal"]}</b></li>
                        <li>AI Confidence: <b>{pred["today"]["confidence"]:.1%}</b></li>
                    </ul>
                    </span>
                </div>
                ''', unsafe_allow_html=True)
                # -------------------------------
                
                # --- ADVANCED ANALYST DATA (Consolidated to resolve confusion) ---
                with st.expander("🛠️ Advanced Analyst Data & Market Context", expanded=False):
                    st.subheader("🤖 Model Accuracy")
                    mc1, mc2, mc3 = st.columns(3)
                    if "Intraday" in pred_mode:
                        mc1.metric("15-Min Accuracy", f"{metrics['d1_acc']:.1%}")
                        mc2.metric("30-Min Accuracy", f"{metrics['d2_acc']:.1%}")
                        mc3.metric("60-Min Accuracy", f"{metrics['d4_acc']:.1%}")
                    else:
                        mc1.metric("1-Day Accuracy", f"{metrics['d1_acc']:.1%}")
                        mc2.metric("2-Day Accuracy", f"{metrics['d2_acc']:.1%}")
                        mc3.metric("3-Day Accuracy", f"{metrics.get('d3_acc', 0):.1%}")

                    if scored_news:
                        st.subheader("📰 News Sentiment Details")
                        sl = "🟢 Bullish" if sent>0.2 else "🔴 Bearish" if sent<-0.2 else "🟡 Neutral"
                        st.markdown(f"**Current Sentiment Bias: {sl} ({sent:+.2f})**")
                        for n in scored_news[:6]:
                            scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(n['label'],'sentiment-neu')
                            st.markdown(f'<div class="news-card"><span class="{scls}">[{n["label"].upper()}]</span> '
                                        f'{n["title"]}</div>', unsafe_allow_html=True)
                
                
                # --- FINAL MASTER VERDICT ---
                if pred:
                    # --- UNIFIED INTELLIGENCE LOGIC ---
                    res = detect_candle_pattern(df_run.tail(3))
                    pat_l = res["pattern"].lower()
                    
                    ai_s = pred["today"]["signal"]
                    tv_str = "BUY" if tv_sentiment > 0.1 else "SELL" if tv_sentiment < -0.1 else "NEUTRAL"
                    pat_str = "BULLISH" if ('bullish' in pat_l or 'hammer' in pat_l) else "BEARISH" if ('bearish' in pat_l or 'star' in pat_l) else "NEUTRAL"
                    news_str = "BULLISH" if sent > 0.2 else "BEARISH" if sent < -0.2 else "NEUTRAL"
                    
                    ut_last = df_run.iloc[-1]
                    ut_s = "BUY" if ut_last['Close'] > ut_last['UT_Trail'] else "SELL"

                    ai_pts = 4 if ai_s == 'BUY' else -4 if ai_s == 'SELL' else 0
                    tv_pts = 2 if tv_str == "BUY" else -2 if tv_str == "SELL" else 0
                    ut_pts = 2 if ut_s == "BUY" else -2
                    pat_pts = 1.5 if pat_str == "BULLISH" else -1.5 if pat_str == "BEARISH" else 0
                    news_pts = 1.5 if news_str == "BULLISH" else -1.5 if news_str == "BEARISH" else 0
                    
                    score = ai_pts + tv_pts + ut_pts + pat_pts + news_pts
                    
                    # Calculate Agreement Score (How many point in same direction?)
                    total_indicators = 5
                    pointing_up = sum([1 for p in [ai_pts, tv_pts, ut_pts, pat_pts, news_pts] if p > 0])
                    pointing_down = sum([1 for p in [ai_pts, tv_pts, ut_pts, pat_pts, news_pts] if p < 0])
                    agreement_pct = (max(pointing_up, pointing_down) / total_indicators) * 100
                    
                    # Target Verdict Fix
                    if score >= 4.5: fin_sig, fin_col, act_desc = "POWERFUL BUY 🚀", "#00b386", "All major systems agree. High probability trend detected."
                    elif score >= 2.0: fin_sig, fin_col, act_desc = "BUY 📈", "#10b981", "Bullish bias dominates. Consider entry on dips."
                    elif score <= -4.5: fin_sig, fin_col, act_desc = "POWERFUL SELL 💥", "#eb5b3c", "Unanimous bearish signals. Strong downward pressure."
                    elif score <= -2.0: fin_sig, fin_col, act_desc = "SELL 📉", "#ef4444", "Bearish signals outweigh. Avoid buying."
                    else: fin_sig, fin_col, act_desc = "WAIT / HOLD ⚖️", "#f59e0b", "Signals are conflicting. No clear trade advantage right now."

                    # Track Drivers
                    drivers = []
                    if abs(ai_pts) >= 4: drivers.append(f"AI Math ({ai_s})")
                    if abs(tv_pts) >= 2: drivers.append(f"TV Technicals ({tv_str})")
                    if abs(ut_pts) >= 2: drivers.append(f"UT Bot ({ut_s})")
                    if abs(pat_pts) >= 1.5: drivers.append(f"Chart Pattern ({pat_str})")
                    if abs(news_pts) >= 1.5: drivers.append(f"News Sentiment ({news_str})")
                    driver_text = " + ".join(drivers) if drivers else "Mixed Indicators"

                    # Conflict Insight
                    if agreement_pct <= 50:
                        insight = "⚠️ <b>High Conflict</b>: Mathematical Trend vs Market Sentiment are fighting. Wait for a clear breakout."
                    elif (ai_pts > 0 and news_pts < 0):
                        insight = "💡 <b>AI vs News</b>: The numbers are bullish, but current news headlines suggest caution."
                    elif (ai_pts < 0 and news_pts > 0):
                        insight = "💡 <b>AI vs News</b>: Math is bearish, but news sentiment is attempting to push back."
                    else:
                        insight = "✅ <b>Trend Sync</b>: Signals are currently in harmony for this direction."

                    # Fixed: Removed leading spaces to prevent Markdown code block interpretation
                    consensus_html = (
f'<div style="background: {fin_col}15; border: 2px solid {fin_col}; padding: 35px; border-radius: 20px; margin: 30px 0; text-align: center; border-left: 10px solid {fin_col};">'
f'<div style="background:{fin_col}; color:white; padding:4px 15px; border-radius:30px; display:inline-block; font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:10px;">Supreme Intelligence Verdict</div>'
f'<h1 style="margin:10px 0; color:{fin_col}; font-size: 4rem; font-weight: 900; letter-spacing:-1px;">{fin_sig}</h1>'
f'<div style="font-size:1.3rem; color:#e2e8f0; font-weight:600;">{act_desc}</div>'

f'<div style="display:flex; justify-content:center; gap:20px; margin:25px 0;">'
f'<div style="background:rgba(255,255,255,0.05); padding:15px; border-radius:15px; border:1px solid rgba(255,255,255,0.1); flex:1;">'
f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Agreement Level</div>'
f'<div style="font-size:1.4rem; color:#cbd5e1; font-weight:800;">{agreement_pct:.0%}%</div>'
f'</div>'
f'<div style="background:rgba(255,255,255,0.05); padding:15px; border-radius:15px; border:1px solid rgba(255,255,255,0.1); flex:2;">'
f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Dominant Driver</div>'
f'<div style="font-size:1rem; color:#cbd5e1; font-weight:700;">{driver_text}</div>'
f'</div>'
f'</div>'

f'<div style="background:rgba(0,0,0,0.3); padding:15px; border-radius:12px; border:1px dashed rgba(255,255,255,0.2); text-align:left;">'
f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; margin-bottom:5px;">System Insight</div>'
f'<div style="color:#e2e8f0; font-size:0.95rem;">{insight}</div>'
f'</div>'
f'</div>'
)
                    st.markdown(consensus_html, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Not enough historical data for AI prediction (need 60+ days). Showing live price and chart only.")
                st.plotly_chart(build_candle_chart(df.tail(60), symbol), use_container_width=True)
        else:
            st.warning("⚠️ This stock is too new or symbol not found for AI prediction. Showing live price only.")
            if df is not None and not df.empty:
                st.plotly_chart(build_candle_chart(df, symbol), use_container_width=True)



# ── PAGE: Market News ─────────────────────────────────────────────────────
def page_news():
    st.subheader("📰 Market News — Live Aggregation")
    
    # NEW: TradingView News Timeline
    st.markdown('<div class="section-head">🔥 TradingView Market Timeline</div>', unsafe_allow_html=True)
    st.components.v1.html(build_tradingview_market_news(), height=600)
    st.write("")
    
    st.markdown('<div class="section-head">📰 MoneyControl & Business News</div>', unsafe_allow_html=True)
    news = fetch_market_news("Indian Stock Market Nifty Sensex Latest News")
    _, scored = analyze_news(news)
    
    if scored:
        for n in scored:
            scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(n['label'],'sentiment-neu')
            url = n.get('url','#')
            st.markdown(f'<div class="news-card"><span class="{scls}">[{n["label"].upper()}]</span> '
                f'<a href="{url}" target="_blank" style="color:#e2e8f0;text-decoration:none">{n["title"]}</a></div>',
                unsafe_allow_html=True)
    else:
        st.info("No fresh news available at the moment.")
        
    st.markdown("---")
    st.markdown("🔗 **Direct Links to Market Exchanges:**")
    st.link_button("📊 NSE India", "https://www.nseindia.com/")
    st.link_button("💼 BSE India", "https://www.bseindia.com/")
    st.link_button("📰 MoneyControl", "https://www.moneycontrol.com/")


# ── PAGE: All Stocks ──────────────────────────────────────────────────────
def page_all_stocks():
    st.subheader("📊 All Stocks — Live Prices")
    category = st.selectbox("Select Category", list(DASHBOARD_CATEGORIES.keys()))
    symbols = DASHBOARD_CATEGORIES[category]
    # Remove duplicates
    symbols = list(dict.fromkeys(symbols))
    
    rows = []
    prog = st.progress(0, text="Loading...")
    for i, sym in enumerate(symbols):
        prog.progress((i+1)/len(symbols), text=f"Fetching {sym}...")
        info = get_price_info(sym, 5)
        if info:
            rows.append({'Stock': sym, 'Price': f"{info['currency']}{info['price']:,.2f}",
                'Change': f"{info['change']:+.2f}", 'Change%': f"{info['pct']:+.2f}%",
                'Volume': f"{info['volume']:,}"})
    prog.empty()

    if rows:
        rdf = pd.DataFrame(rows)
        def styl(val):
            if '+' in str(val): return 'color: #10b981; font-weight: bold'
            elif '-' in str(val): return 'color: #ef4444; font-weight: bold'
            return ''
        st.dataframe(rdf.style.map(styl, subset=['Change','Change%']),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("No data available")


# ── PAGE: Top Movers ──────────────────────────────────────────────────────
def page_top_movers():
    st.subheader("🏆 Top Movers Today")
    all_syms = ['RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK','SBIN','ITC','BHARTIARTL',
                'TATASTEEL','TATAMOTORS','TATAPOWER','TATACONSUM','TATAELXSI','TATACOMM',
                'WIPRO','MARUTI','BAJFINANCE','BAJFINSV','BAJAJ-AUTO',
                'ADANIENT','ADANIGREEN','ADANIPORTS','ADANIPOWER',
                'JSWSTEEL','VEDL','NIPPON','COALINDIA','HINDALCO','NMDC','SAIL','JINDALSTEL',
                'NTPC','ONGC','TECHM','HCLTECH','SUNPHARMA','TITAN','TRENT',
                'AXISBANK','LT','KOTAKBANK','M&M','HEROMOTOCO','DRREDDY','CIPLA',
                'NESTLEIND','HINDUNILVR','BRITANNIA','DABUR','BOSCHLTD','SIEMENS',
                'SHRIRAMFIN','EICHERMOT','BANKBARODA','PNB','MUTHOOTFIN','HAVELLS',
                'VOLTAS','DLF','GODREJPROP','IRCTC','IRFC','RVNL',
                'HAL','BEL','ZOMATO','PAYTM','POLYCAB','DIXON',
                'LUPIN','BIOCON','GLENMARK','PIDILITIND','SRF',
                'POWERGRID','BPCL','IOC','GAIL','NHPC','PFC','RECLTD',
                'TVSMOTOR','ASHOKLEY','MRF','APOLLOTYRE',
                'CHOLAFIN','MANAPPURAM','LICHSGFIN','LICI','SBILIFE','HDFCLIFE',
                'MARICO','COLPAL','DMART','JUBLFOOD',
                'ABB','CROMPTON','KEI','THERMAX']
    data = []
    prog = st.progress(0, text="Scanning market...")
    for i, sym in enumerate(all_syms):
        prog.progress((i+1)/len(all_syms), text=f"{sym}...")
        info = get_price_info(sym, 5)
        if info: data.append(info)
    prog.empty()

    if not data:
        st.warning("No data"); return

    gainers = sorted([d for d in data if d['pct']>0], key=lambda x:-x['pct'])[:15]
    losers = sorted([d for d in data if d['pct']<0], key=lambda x:x['pct'])[:15]

    t1, t2 = st.tabs(["🟢 Top Gainers", "🔴 Top Losers"])
    with t1:
        if gainers:
            for g in gainers:
                st.markdown(f"""<div class="stock-card" style="display:flex;justify-content:space-between;align-items:center">
                    <div><div class="name">{g['symbol']}</div></div>
                    <div style="text-align:right"><div class="price" style="font-size:1rem">{g['currency']}{g['price']:,.2f}</div>
                    <div class="change-up">{g['change']:+.2f} ({g['pct']:+.2f}%)</div></div>
                </div>""", unsafe_allow_html=True)
    with t2:
        if losers:
            for l in losers:
                st.markdown(f"""<div class="stock-card" style="display:flex;justify-content:space-between;align-items:center">
                    <div><div class="name">{l['symbol']}</div></div>
                    <div style="text-align:right"><div class="price" style="font-size:1rem">{l['currency']}{l['price']:,.2f}</div>
                    <div class="change-down">{l['change']:+.2f} ({l['pct']:+.2f}%)</div></div>
                </div>""", unsafe_allow_html=True)


# ── PAGE: Stock Screener ──────────────────────────────────────────────────
def page_screener():
    st.subheader("🚀 Power Screener — Find Breakout Stocks")
    st.caption("Scan the market based on technical indicators and volume spikes to find high-probability trades.")
    
    with st.expander("🛠️ Screener Filters (Technical & Fundamentals)", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            rsi_filter = st.selectbox("RSI Signal", ["None", "Oversold (<35)", "Bullish (>60)", "Overbought (>70)"])
        with c2:
            vol_filter = st.selectbox("Volume Spike", ["None", "High (>2x Avg)", "Extreme (>5x Avg)"])
        with c3:
            pe_filter = st.selectbox("P/E Ratio", ["Any", "Under 15", "Under 25", "Under 40"])
        with c4:
            pat_filter = st.selectbox("Candlestick Pattern", ["Any", "Bullish Hammer", "Bullish Engulfing", "Bearish Engulfing", "Doji"])
        with c5:
            ut_filter = st.selectbox("UT Bot Alert", ["Any", "BUY Signal", "SELL Signal"])
            
    if st.button("🔍 Start Market Scan", use_container_width=True):
        # We screen the Nifty 50 and Top Growth stocks for speed
        screen_list = list(dict.fromkeys(DASHBOARD_CATEGORIES['🏛️ Indices'] + 
                                       DASHBOARD_CATEGORIES['🔵 Tata Group'] + 
                                       DASHBOARD_CATEGORIES['🏢 Adani Group'] +
                                       DASHBOARD_CATEGORIES['💻 IT & Software'] +
                                       DASHBOARD_CATEGORIES['🏦 Public Banks'] +
                                       DASHBOARD_CATEGORIES['🏧 Private Banks'] +
                                       DASHBOARD_CATEGORIES['💊 Pharma'] +
                                       DASHBOARD_CATEGORIES['🛒 FMCG'] +
                                       DASHBOARD_CATEGORIES['🚗 Auto']))
        
        # Limit to 100 stocks for performance if needed, but here we try all for thoroughness
        screen_list = screen_list[:120] 
        
        matches = []
        prog = st.progress(0, text="Scanning Market Pulse...")
        
        for i, sym in enumerate(screen_list):
            prog.progress((i+1)/len(screen_list), text=f"Analyzing {sym}...")
            df, mapped = fetch_stock(sym, 30)
            if df is not None and len(df) > 14:
                # Calculate RSI
                closes = df['Close'].dropna().astype(float).values
                rsi = AIEngine._rsi(closes)
                
                # Calculate Volume Spike
                vols = df['Volume'].dropna().values
                avg_vol = np.mean(vols[:-1]) if len(vols) > 1 else 1
                curr_vol = vols[-1]
                vol_ratio = float(curr_vol / avg_vol) if avg_vol > 0 else 0.0
                
                # Detect Pattern & UT Bot
                pat_res = detect_candle_pattern(df)
                df = calculate_ut_bot(df)
                ut_last = df.iloc[-1]
                ut_s = "BUY" if ut_last['Close'] > ut_last['UT_Trail'] else "SELL"
                
                # Apply Filters
                pass_rsi = True
                if rsi_filter == "Oversold (<35)": pass_rsi = rsi < 35
                elif rsi_filter == "Bullish (>60)": pass_rsi = rsi > 60
                elif rsi_filter == "Overbought (>70)": pass_rsi = rsi > 70
                
                pass_vol = True
                if vol_filter == "High (>2x Avg)": pass_vol = vol_ratio > 2
                elif vol_filter == "Extreme (>5x Avg)": pass_vol = vol_ratio > 5
                
                pass_pat = True
                if pat_filter != "Any":
                    pass_pat = pat_filter.lower() in pat_res['pattern'].lower()

                pass_ut = True
                if ut_filter == "BUY Signal": pass_ut = ut_s == "BUY"
                elif ut_filter == "SELL Signal": pass_ut = ut_s == "SELL"

                # Apply PE Filter
                pass_pe = True
                f_stats = None # Initialize to prevent NameError
                if pe_filter != "Any":
                    f_stats = fetch_fundamentals(mapped)
                    curr_pe = f_stats['pe'] if f_stats else 0
                    if curr_pe > 0: # Only filter if PE data exists
                        if pe_filter == "Under 15": pass_pe = curr_pe < 15
                        elif pe_filter == "Under 25": pass_pe = curr_pe < 25
                        elif pe_filter == "Under 40": pass_pe = curr_pe < 40
                
                if pass_rsi and pass_vol and pass_pat and pass_pe and pass_ut:
                    info = get_price_info(sym, 2)
                    if info:
                        # NEW: Market News Collector for the matched stock
                        news_items = fetch_market_news(f"{sym} share stock news")
                        lat_news = news_items[0]['title'] if news_items else "No recent news"
                        
                        matches.append({
                            'Stock': sym,
                            'Price': f"{info['currency']}{info['price']:,.2f}",
                            'Change%': f"{info['pct']:+.2f}%",
                            'UT Bot': f"{ut_s} 🤖",
                            'RSI': f"{rsi:.1f}",
                            'P/E': f"{f_stats['pe']:.1f}" if (pe_filter != "Any" and f_stats) else 
                                   f"{fetch_fundamentals(mapped)['pe']:.1f}" if fetch_fundamentals(mapped) else "N/A",
                            'Vol': f"{vol_ratio:.1f}x",
                            'Pattern': pat_res['pattern'],
                            'Latest News': lat_news
                        })
        prog.empty()
        
        if matches:
            st.success(f"✅ Found {len(matches)} stocks matching your criteria!")
            m_df = pd.DataFrame(matches)
            
            # Stylized display
            def styl_screener(val):
                if any(x in str(val) for x in ['Bullish', 'Hammer', '+', 'Positive']): return 'color: #10b981; font-weight: bold'
                if any(x in str(val) for x in ['Bearish', '-', 'Negative']): return 'color: #ef4444; font-weight: bold'
                return ''
                
            st.dataframe(m_df.style.map(styl_screener), use_container_width=True, hide_index=True)
            
            # Quick Insight
            st.info("💡 **Market News Collector**: The 'Latest News catalyst' column shows the real-time reason for the price action. Combine technical signals (RSI/Volume) with these news headlines for higher accuracy.")
        else:
            st.warning("❌ No stocks found matching these exact filters. Try loosening the criteria.")


# ── PAGE: Sector View ────────────────────────────────────────────────────
def page_sector_view():
    st.subheader("📈 Sector-wise Performance")
    for sector, syms in DASHBOARD_CATEGORIES.items(): 
        if sector == '🏛️ Indices': continue
        unique_syms = list(dict.fromkeys(syms))[:5]
        with st.expander(f"{sector}"):
            cols = st.columns(len(unique_syms))
            for i, sym in enumerate(unique_syms):
                info = get_price_info(sym, 5)
                if info:
                    cls = 'change-up' if info['change']>=0 else 'change-down'
                    with cols[i]:
                        st.markdown(f"""<div class="stock-card">
                            <div class="name">{sym}</div>
                            <div class="price" style="font-size:1rem">{info['currency']}{info['price']:,.2f}</div>
                            <div class="{cls}">{info['pct']:+.2f}%</div>
                        </div>""", unsafe_allow_html=True)


if __name__ == "__main__": 
    main()
