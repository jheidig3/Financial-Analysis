import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Quantitative Options & Trading Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom UI Styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""",
    unsafe_allow_html=True,
)


# QUANTITATIVE MATH & ANALYTICS UTILITIES



def norm_cdf(x):
  """Cumulative distribution function for standard normal distribution."""
  return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x):
  """Probability density function for standard normal distribution."""
  return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def black_scholes(S, K, T, r, sigma, option_type='call'):
  """Calculates Black-Scholes option price and Greeks."""
  if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
    return {'price': 0.0, 'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

  d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)

  if option_type == 'call':
    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
  else:
    price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    delta = norm_cdf(d1) - 1.0

  gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
  vega = (S * norm_pdf(d1) * math.sqrt(T)) / 100.0  # 1% IV shift

  if option_type == 'call':
    theta = (
        -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * norm_cdf(d2)
    ) / 365.0
  else:
    theta = (
        -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * norm_cdf(-d2)
    ) / 365.0

  return {
      'price': max(0.0, price),
      'delta': delta,
      'gamma': gamma,
      'theta': theta,
      'vega': vega,
  }


def generate_option_chain(S, r, iv, dte_days):
  """Generates a synthetic options chain with computed Black-Scholes Greeks."""
  T = max(dte_days / 365.0, 0.001)
  strikes = np.linspace(S * 0.75, S * 1.25, 31)
  chain = []

  for K in strikes:
    K = round(K, 2)
    call = black_scholes(S, K, T, r, iv, 'call')
    put = black_scholes(S, K, T, r, iv, 'put')

    chain.append({
        'Strike': K,
        'Call Price': round(call['price'], 2),
        'Call Delta': round(call['delta'], 3),
        'Call Theta': round(call['theta'], 3),
        'Put Price': round(put['price'], 2),
        'Put Delta': round(put['delta'], 3),
        'Put Theta': round(put['theta'], 3),
    })
  return pd.DataFrame(chain)


def run_delta_options_screener(
    universe_df,
    target_delta_min,
    target_delta_max,
    option_type,
    min_pop,
    min_yield,
):
  """Automated options selling screener filtering contracts by target Delta, PoP, and Yield."""
  results = []
  for _, row in universe_df.iterrows():
    S = row['Spot']
    iv = row['IV']
    ticker = row['Ticker']
    r = 0.045
    dte = 45
    T = dte / 365.0

    strikes = np.linspace(S * 0.70, S * 1.30, 61)
    for K in strikes:
      K = round(K, 2)
      if option_type == 'Put' and K >= S:
        continue
      if option_type == 'Call' and K <= S:
        continue

      opt_type = 'put' if option_type == 'Put' else 'call'
      bs = black_scholes(S, K, T, r, iv, opt_type)
      abs_delta = abs(bs['delta'])

      if target_delta_min <= abs_delta <= target_delta_max:
        pop = (1 - abs_delta) if option_type == 'Put' else (1 - bs['delta'])
        premium = bs['price']

        capital_required = K * 100 if option_type == 'Put' else S * 100
        premium_income = premium * 100
        return_on_cap = (
            (premium_income / capital_required) * (365 / dte) * 100
            if capital_required > 0
            else 0
        )

        if pop >= min_pop and return_on_cap >= min_yield:
          # Compute 1-6 Option Seller Score
          seller_score = 6 if abs_delta <= 0.25 and return_on_cap >= 15 else 4

          results.append({
              'Ticker': ticker,
              'Spot Price': f'${S:.2f}',
              'Option Type': option_type,
              'Strike': f'${K:.2f}',
              'DTE': dte,
              'Delta': f"{bs['delta']:.3f}",
              'Abs Delta': round(abs_delta, 3),
              'Premium ($)': f'${premium:.2f}',
              'PoP (%)': f'{pop * 100:.1f}%',
              'Ann. Yield (%)': f'{return_on_cap:.1f}%',
              'Seller Score': seller_score,
          })
  return pd.DataFrame(results)


# Sample Asset Universe
DEFAULT_UNIVERSE = pd.DataFrame([
    {'Ticker': 'NVDA', 'Spot': 135.00, 'IV': 0.45, 'Sector': 'Semiconductors'},
    {'Ticker': 'AAPL', 'Spot': 225.00, 'IV': 0.24, 'Sector': 'Technology'},
    {'Ticker': 'AMZN', 'Spot': 185.00, 'IV': 0.32, 'Sector': 'Consumer Cyclical'},
    {'Ticker': 'MSFT', 'Spot': 420.00, 'IV': 0.22, 'Sector': 'Technology'},
    {'Ticker': 'GOOGL', 'Spot': 165.00, 'IV': 0.28, 'Sector': 'Communication'},
    {'Ticker': 'BE', 'Spot': 24.50, 'IV': 0.65, 'Sector': 'Clean Energy'},
    {'Ticker': 'INTC', 'Spot': 22.00, 'IV': 0.48, 'Sector': 'Semiconductors'},
    {'Ticker': 'SPY', 'Spot': 560.00, 'IV': 0.15, 'Sector': 'ETF'},
])


# SIDEBAR CONTROLS

st.sidebar.title('⚙️ Engine Controls')
st.sidebar.markdown('---')

selected_ticker = st.sidebar.selectbox(
    'Select Core Asset Ticker', DEFAULT_UNIVERSE['Ticker'].tolist(), index=0
)
spot_override = st.sidebar.number_input(
    'Spot Price ($)',
    value=float(
        DEFAULT_UNIVERSE.loc[
            DEFAULT_UNIVERSE['Ticker'] == selected_ticker, 'Spot'
        ].values[0]
    ),
    step=1.0,
)
iv_override = (
    st.sidebar.slider(
        'Implied Volatility (IV %)',
        min_value=10,
        max_value=120,
        value=int(
            DEFAULT_UNIVERSE.loc[
                DEFAULT_UNIVERSE['Ticker'] == selected_ticker, 'IV'
            ].values[0]
            * 100
        ),
        step=1,
    )
    / 100.0
)
r_rate = (
    st.sidebar.number_input('Risk-Free Rate (%)', value=4.5, step=0.1) / 100.0
)

st.sidebar.markdown('---')
st.sidebar.subheader('🎯 Automated Delta Filter')
filter_delta_range = st.sidebar.slider(
    'Target Delta Selling Window', 0.05, 0.40, (0.15, 0.30), step=0.01
)
filter_option_type = st.sidebar.radio(
    'Screener Strategy', ['Put', 'Call'], index=0
)
min_pop_threshold = (
    st.sidebar.slider(
        'Min Probability of Profit (PoP %)', 50, 95, 70, step=5
    )
    / 100.0
)
min_ann_yield = st.sidebar.slider(
    'Min Annualized Yield (%)', 5, 50, 15, step=1
)


# MAIN INTERFACE & NAVIGATION


st.markdown(
    '<div class="main-header">Quantitative Options & Trading Analytics'
    ' Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">LEAPS Buying Scanner (1+ Year Expiration), Option'
    ' Seller Engine, Greeks Matrix, and Congressional Tracker</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🚀 LEAPS Scanner (1+ Yr)',
    '💰 Option Seller Engine',
    '📊 Option Chain & Greeks',
    '📈 Payoff Analyzer',
    '🏛 Congressional Tracker',
])

# ------------------------------------------
# TAB 1: LEAPS SCANNER (1+ YEAR / 0.80+ DELTA)
# ------------------------------------------
with tab1:
    st.subheader('🚀 LEAPS Call Scanner (Minimum 1-Year Expiration Target)')
    st.markdown(
        'Scans for deep in-the-money (ITM) Call options with **$\ge 365$ DTE**'
        ' and **$\ge 0.80$ Delta** for stock replacement strategies.'
    )

    # Compute LEAPS candidates for selected ticker
    leaps_dte = 450  # ~15 months out
    T_leaps = leaps_dte / 365.0
    strikes = np.linspace(spot_override * 0.60, spot_override * 0.95, 12)

    leaps_list = []
    for k in strikes:
      k = round(k, 1)
      bs = black_scholes(spot_override, k, T_leaps, r_rate, iv_override, 'call')
      if bs['delta'] >= 0.80:
        breakeven = k + bs['price']
        extrinsic = bs['price'] - max(0, spot_override - k)
        leaps_list.append({
            'Strike': f'${k:.2f}',
            'Target Expiration': (
                datetime.now() + timedelta(days=leaps_dte)
            ).strftime('%Y-%m-%d'),
            'DTE': leaps_dte,
            'Est. Premium': f"${bs['price']:.2f}",
            'Delta': f"{bs['delta']:.3f}",
            'Breakeven': f'${breakeven:.2f}',             'Extrinsic Cost': f'${extrinsic:.2f}',
            'LEAPS Score': 6 if bs['delta'] >= 0.82 else 5,
        })

    leaps_df = pd.DataFrame(leaps_list)

    if not leaps_df.empty:
      st.info(
          f'💡 **Suggested Primary LEAPS Candidate for {selected_ticker}:**'
          f' **{leaps_df.iloc[0]["Strike"]} Call** | Target Expiration:'
          f' `{leaps_df.iloc[0]["Target Expiration"]}` ({leaps_dte} DTE) |'
          f' Delta: `{leaps_df.iloc[0]["Delta"]}`'
      )

      st.dataframe(leaps_df, use_container_width=True, hide_index=True)
    else:
      st.warning('No contracts matching $\ge 0.80$ Delta parameters.')

# ------------------------------------------
# TAB 2: OPTION SELLER ENGINE (1-6 SCORING)
# ------------------------------------------
with tab2:
    st.subheader('💰 Option Seller & Premium Analyzer')
    st.markdown(
        'Evaluates market conditions against a **1 to 6 Score Scale** to identify'
        ' optimal environments for writing Cash-Secured Puts or Covered Calls.'
    )

    screener_df = run_delta_options_screener(
        DEFAULT_UNIVERSE,
        filter_delta_range[0],
        filter_delta_range[1],
        filter_option_type,
        min_pop_threshold,
        min_ann_yield,
    )

    if not screener_df.empty:

      def style_score(val):
        if val >= 6:
          return 'background-color: #16A34A; color: white; font-weight: bold;'
        elif 4 <= val <= 5:
          return 'background-color: #FEF08A; color: #854D0E; font-weight: bold;'
        else:
          return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'

      st.dataframe(
          screener_df.style.map(style_score, subset=['Seller Score']),
          use_container_width=True,
          hide_index=True,
      )
    else:
      st.warning(
          'No option selling opportunities match current filter constraints.'
      )

# ------------------------------------------
# TAB 3: OPTION CHAIN & GREEKS
# ------------------------------------------
with tab3:
    st.subheader(f'📊 Synthetic Option Chain & Greeks: {selected_ticker}')
    dte_selection = st.slider(
        'Days to Expiration (DTE)',
        min_value=7,
        max_value=120,
        value=45,
        step=1,
    )

    chain_df = generate_option_chain(
        spot_override, r_rate, iv_override, dte_selection
    )
    st.dataframe(chain_df, use_container_width=True, height=350)

# ------------------------------------------
# TAB 4: PAYOFF ANALYZER
# ------------------------------------------
with tab4:
    st.subheader('📈 Payoff & Strategy Engine')
    strategy = st.selectbox(
        'Strategy', ['Short Cash-Secured Put', 'Covered Call', 'Long Call LEAPS']
    )

    price_range = np.linspace(spot_override * 0.70, spot_override * 1.30, 100)
    payoff = []

    if strategy == 'Short Cash-Secured Put':
      put_strike = round(spot_override * 0.95, 1)
      premium = 2.50
      for p in price_range:
        payoff.append((premium - max(0, put_strike - p)) * 100)
      title = f'Short Put Payoff (${put_strike} Strike)'
    elif strategy == 'Covered Call':
      call_strike = round(spot_override * 1.05, 1)
      premium = 3.00
      for p in price_range:
        payoff.append(
            ((p - spot_override) + premium - max(0, p - call_strike)) * 100
        )
      title = f'Covered Call Payoff (${call_strike} Strike)'
    else:
      call_strike = round(spot_override * 0.80, 1)
      premium = spot_override * 0.25
      for p in price_range:
        payoff.append((max(0, p - call_strike) - premium) * 100)
      title = f'Long LEAPS Call Payoff (${call_strike} Strike)'

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=payoff,
            mode='lines',
            line=dict(width=3, color='blue'),
        )
    )
    fig.add_hline(y=0, line_color='gray')
    fig.update_layout(
        title=title,
        xaxis_title='Stock Price at Expiration ($)',
        yaxis_title='P&L ($)',
        template='plotly_white',
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 5: CONGRESSIONAL TRACKER
# ------------------------------------------
with tab5:
    st.subheader('🏛 Congressional Disclosure Tracker')
    trades_data = pd.DataFrame([
        {
            'Filing Date': '2026-08-21',
            'Member': 'Nancy Pelosi',
            'Ticker': 'BE',
            'Asset Class': 'Call Options',
            'Details': '100 Call Options, $100 Strike',
            'Amount': '$500K - $1M',
        },
        {
            'Filing Date': '2026-08-21',
            'Member': 'Nancy Pelosi',
            'Ticker': 'INTC',
            'Asset Class': 'Common Stock',
            'Details': 'Purchased 10,000 shares',
            'Amount': '$500K - $1M',
        },
        {
            'Filing Date': '2026-06-23',
            'Member': 'Nancy Pelosi',
            'Ticker': 'UBER',
            'Asset Class': 'Call Options',
            'Details': '200 Call Options, Deep ITM',
            'Amount': '$500K - $1M',
        },
    ])
    st.dataframe(trades_data, use_container_width=True, hide_index=True)

st.markdown('---')
st.caption('Quantitative Options Analytics Platform | Built with Streamlit & Plotly.')

import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Quantitative Options & Trading Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom UI Styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""",
    unsafe_allow_html=True,
)


# QUANTITATIVE MATH & ANALYTICS UTILITIES



def norm_cdf(x):
  """Cumulative distribution function for standard normal distribution."""
  return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x):
  """Probability density function for standard normal distribution."""
  return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def black_scholes(S, K, T, r, sigma, option_type='call'):
  """Calculates Black-Scholes option price and Greeks."""
  if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
    return {'price': 0.0, 'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

  d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)

  if option_type == 'call':
    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
  else:
    price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    delta = norm_cdf(d1) - 1.0

  gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
  vega = (S * norm_pdf(d1) * math.sqrt(T)) / 100.0  # 1% IV shift

  if option_type == 'call':
    theta = (
        -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * norm_cdf(d2)
    ) / 365.0
  else:
    theta = (
        -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * norm_cdf(-d2)
    ) / 365.0

  return {
      'price': max(0.0, price),
      'delta': delta,
      'gamma': gamma,
      'theta': theta,
      'vega': vega,
  }


def generate_option_chain(S, r, iv, dte_days):
  """Generates a synthetic options chain with computed Black-Scholes Greeks."""
  T = max(dte_days / 365.0, 0.001)
  strikes = np.linspace(S * 0.75, S * 1.25, 31)
  chain = []

  for K in strikes:
    K = round(K, 2)
    call = black_scholes(S, K, T, r, iv, 'call')
    put = black_scholes(S, K, T, r, iv, 'put')

    chain.append({
        'Strike': K,
        'Call Price': round(call['price'], 2),
        'Call Delta': round(call['delta'], 3),
        'Call Theta': round(call['theta'], 3),
        'Put Price': round(put['price'], 2),
        'Put Delta': round(put['delta'], 3),
        'Put Theta': round(put['theta'], 3),
    })
  return pd.DataFrame(chain)


def run_delta_options_screener(
    universe_df,
    target_delta_min,
    target_delta_max,
    option_type,
    min_pop,
    min_yield,
):
  """Automated options selling screener filtering contracts by target Delta, PoP, and Yield."""
  results = []
  for _, row in universe_df.iterrows():
    S = row['Spot']
    iv = row['IV']
    ticker = row['Ticker']
    r = 0.045
    dte = 45
    T = dte / 365.0

    strikes = np.linspace(S * 0.70, S * 1.30, 61)
    for K in strikes:
      K = round(K, 2)
      if option_type == 'Put' and K >= S:
        continue
      if option_type == 'Call' and K <= S:
        continue

      opt_type = 'put' if option_type == 'Put' else 'call'
      bs = black_scholes(S, K, T, r, iv, opt_type)
      abs_delta = abs(bs['delta'])

      if target_delta_min <= abs_delta <= target_delta_max:
        pop = (1 - abs_delta) if option_type == 'Put' else (1 - bs['delta'])
        premium = bs['price']

        capital_required = K * 100 if option_type == 'Put' else S * 100
        premium_income = premium * 100
        return_on_cap = (
            (premium_income / capital_required) * (365 / dte) * 100
            if capital_required > 0
            else 0
        )

        if pop >= min_pop and return_on_cap >= min_yield:
          # Compute 1-6 Option Seller Score
          seller_score = 6 if abs_delta <= 0.25 and return_on_cap >= 15 else 4

          results.append({
              'Ticker': ticker,
              'Spot Price': f'${S:.2f}',
              'Option Type': option_type,
              'Strike': f'${K:.2f}',
              'DTE': dte,
              'Delta': f"{bs['delta']:.3f}",
              'Abs Delta': round(abs_delta, 3),
              'Premium ($)': f'${premium:.2f}',
              'PoP (%)': f'{pop * 100:.1f}%',
              'Ann. Yield (%)': f'{return_on_cap:.1f}%',
              'Seller Score': seller_score,
          })
  return pd.DataFrame(results)


# Sample Asset Universe
DEFAULT_UNIVERSE = pd.DataFrame([
    {'Ticker': 'NVDA', 'Spot': 135.00, 'IV': 0.45, 'Sector': 'Semiconductors'},
    {'Ticker': 'AAPL', 'Spot': 225.00, 'IV': 0.24, 'Sector': 'Technology'},
    {'Ticker': 'AMZN', 'Spot': 185.00, 'IV': 0.32, 'Sector': 'Consumer Cyclical'},
    {'Ticker': 'MSFT', 'Spot': 420.00, 'IV': 0.22, 'Sector': 'Technology'},
    {'Ticker': 'GOOGL', 'Spot': 165.00, 'IV': 0.28, 'Sector': 'Communication'},
    {'Ticker': 'BE', 'Spot': 24.50, 'IV': 0.65, 'Sector': 'Clean Energy'},
    {'Ticker': 'INTC', 'Spot': 22.00, 'IV': 0.48, 'Sector': 'Semiconductors'},
    {'Ticker': 'SPY', 'Spot': 560.00, 'IV': 0.15, 'Sector': 'ETF'},
])


# SIDEBAR CONTROLS

st.sidebar.title('⚙️ Engine Controls')
st.sidebar.markdown('---')

selected_ticker = st.sidebar.selectbox(
    'Select Core Asset Ticker', DEFAULT_UNIVERSE['Ticker'].tolist(), index=0
)
spot_override = st.sidebar.number_input(
    'Spot Price ($)',
    value=float(
        DEFAULT_UNIVERSE.loc[
            DEFAULT_UNIVERSE['Ticker'] == selected_ticker, 'Spot'
        ].values[0]
    ),
    step=1.0,
)
iv_override = (
    st.sidebar.slider(
        'Implied Volatility (IV %)',
        min_value=10,
        max_value=120,
        value=int(
            DEFAULT_UNIVERSE.loc[
                DEFAULT_UNIVERSE['Ticker'] == selected_ticker, 'IV'
            ].values[0]
            * 100
        ),
        step=1,
    )
    / 100.0
)
r_rate = (
    st.sidebar.number_input('Risk-Free Rate (%)', value=4.5, step=0.1) / 100.0
)

st.sidebar.markdown('---')
st.sidebar.subheader('🎯 Automated Delta Filter')
filter_delta_range = st.sidebar.slider(
    'Target Delta Selling Window', 0.05, 0.40, (0.15, 0.30), step=0.01
)
filter_option_type = st.sidebar.radio(
    'Screener Strategy', ['Put', 'Call'], index=0
)
min_pop_threshold = (
    st.sidebar.slider(
        'Min Probability of Profit (PoP %)', 50, 95, 70, step=5
    )
    / 100.0
)
min_ann_yield = st.sidebar.slider(
    'Min Annualized Yield (%)', 5, 50, 15, step=1
)


# MAIN INTERFACE & NAVIGATION


st.markdown(
    '<div class="main-header">Quantitative Options & Trading Analytics'
    ' Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">LEAPS Buying Scanner (1+ Year Expiration), Option'
    ' Seller Engine, Greeks Matrix, and Congressional Tracker</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🚀 LEAPS Scanner (1+ Yr)',
    '💰 Option Seller Engine',
    '📊 Option Chain & Greeks',
    '📈 Payoff Analyzer',
    '🏛 Congressional Tracker',
])

# ------------------------------------------
# TAB 1: LEAPS SCANNER (1+ YEAR / 0.80+ DELTA)
# ------------------------------------------
with tab1:
    st.subheader('🚀 LEAPS Call Scanner (Minimum 1-Year Expiration Target)')
    st.markdown(
        'Scans for deep in-the-money (ITM) Call options with **$\ge 365$ DTE**'
        ' and **$\ge 0.80$ Delta** for stock replacement strategies.'
    )

    # Compute LEAPS candidates for selected ticker
    leaps_dte = 450  # ~15 months out
    T_leaps = leaps_dte / 365.0
    strikes = np.linspace(spot_override * 0.60, spot_override * 0.95, 12)

    leaps_list = []
    for k in strikes:
      k = round(k, 1)
      bs = black_scholes(spot_override, k, T_leaps, r_rate, iv_override, 'call')
      if bs['delta'] >= 0.80:
        breakeven = k + bs['price']
        extrinsic = bs['price'] - max(0, spot_override - k)
        leaps_list.append({
            'Strike': f'${k:.2f}',
            'Target Expiration': (
                datetime.now() + timedelta(days=leaps_dte)
            ).strftime('%Y-%m-%d'),
            'DTE': leaps_dte,
            'Est. Premium': f"${bs['price']:.2f}",
            'Delta': f"{bs['delta']:.3f}",
            'Breakeven': f'${breakeven:.2f}',             'Extrinsic Cost': f'${extrinsic:.2f}',
            'LEAPS Score': 6 if bs['delta'] >= 0.82 else 5,
        })

    leaps_df = pd.DataFrame(leaps_list)

    if not leaps_df.empty:
      st.info(
          f'💡 **Suggested Primary LEAPS Candidate for {selected_ticker}:**'
          f' **{leaps_df.iloc[0]["Strike"]} Call** | Target Expiration:'
          f' `{leaps_df.iloc[0]["Target Expiration"]}` ({leaps_dte} DTE) |'
          f' Delta: `{leaps_df.iloc[0]["Delta"]}`'
      )

      st.dataframe(leaps_df, use_container_width=True, hide_index=True)
    else:
      st.warning('No contracts matching $\ge 0.80$ Delta parameters.')

# ------------------------------------------
# TAB 2: OPTION SELLER ENGINE (1-6 SCORING)
# ------------------------------------------
with tab2:
    st.subheader('💰 Option Seller & Premium Analyzer')
    st.markdown(
        'Evaluates market conditions against a **1 to 6 Score Scale** to identify'
        ' optimal environments for writing Cash-Secured Puts or Covered Calls.'
    )

    screener_df = run_delta_options_screener(
        DEFAULT_UNIVERSE,
        filter_delta_range[0],
        filter_delta_range[1],
        filter_option_type,
        min_pop_threshold,
        min_ann_yield,
    )

    if not screener_df.empty:

      def style_score(val):
        if val >= 6:
          return 'background-color: #16A34A; color: white; font-weight: bold;'
        elif 4 <= val <= 5:
          return 'background-color: #FEF08A; color: #854D0E; font-weight: bold;'
        else:
          return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'

      st.dataframe(
          screener_df.style.map(style_score, subset=['Seller Score']),
          use_container_width=True,
          hide_index=True,
      )
    else:
      st.warning(
          'No option selling opportunities match current filter constraints.'
      )

# ------------------------------------------
# TAB 3: OPTION CHAIN & GREEKS
# ------------------------------------------
with tab3:
    st.subheader(f'📊 Synthetic Option Chain & Greeks: {selected_ticker}')
    dte_selection = st.slider(
        'Days to Expiration (DTE)',
        min_value=7,
        max_value=120,
        value=45,
        step=1,
    )

    chain_df = generate_option_chain(
        spot_override, r_rate, iv_override, dte_selection
    )
    st.dataframe(chain_df, use_container_width=True, height=350)

# ------------------------------------------
# TAB 4: PAYOFF ANALYZER
# ------------------------------------------
with tab4:
    st.subheader('📈 Payoff & Strategy Engine')
    strategy = st.selectbox(
        'Strategy', ['Short Cash-Secured Put', 'Covered Call', 'Long Call LEAPS']
    )

    price_range = np.linspace(spot_override * 0.70, spot_override * 1.30, 100)
    payoff = []

    if strategy == 'Short Cash-Secured Put':
      put_strike = round(spot_override * 0.95, 1)
      premium = 2.50
      for p in price_range:
        payoff.append((premium - max(0, put_strike - p)) * 100)
      title = f'Short Put Payoff (${put_strike} Strike)'
    elif strategy == 'Covered Call':
      call_strike = round(spot_override * 1.05, 1)
      premium = 3.00
      for p in price_range:
        payoff.append(
            ((p - spot_override) + premium - max(0, p - call_strike)) * 100
        )
      title = f'Covered Call Payoff (${call_strike} Strike)'
    else:
      call_strike = round(spot_override * 0.80, 1)
      premium = spot_override * 0.25
      for p in price_range:
        payoff.append((max(0, p - call_strike) - premium) * 100)
      title = f'Long LEAPS Call Payoff (${call_strike} Strike)'

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=payoff,
            mode='lines',
            line=dict(width=3, color='blue'),
        )
    )
    fig.add_hline(y=0, line_color='gray')
    fig.update_layout(
        title=title,
        xaxis_title='Stock Price at Expiration ($)',
        yaxis_title='P&L ($)',
        template='plotly_white',
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 5: CONGRESSIONAL TRACKER
# ------------------------------------------
with tab5:
    st.subheader('🏛 Congressional Disclosure Tracker')
    trades_data = pd.DataFrame([
        {
            'Filing Date': '2026-08-21',
            'Member': 'Nancy Pelosi',
            'Ticker': 'BE',
            'Asset Class': 'Call Options',
            'Details': '100 Call Options, $100 Strike',
            'Amount': '$500K - $1M',
        },
        {
            'Filing Date': '2026-08-21',
            'Member': 'Nancy Pelosi',
            'Ticker': 'INTC',
            'Asset Class': 'Common Stock',
            'Details': 'Purchased 10,000 shares',
            'Amount': '$500K - $1M',
        },
        {
            'Filing Date': '2026-06-23',
            'Member': 'Nancy Pelosi',
            'Ticker': 'UBER',
            'Asset Class': 'Call Options',
            'Details': '200 Call Options, Deep ITM',
            'Amount': '$500K - $1M',
        },
    ])
    st.dataframe(trades_data, use_container_width=True, hide_index=True)

st.markdown('---')
st.caption('Quantitative Options Analytics Platform | Built with Streamlit & Plotly.')