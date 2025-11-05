# streamlit_dashboard.py
# Streamlit dashboard for tourist-destination visualisation & route recommendations
# Reads data from /mnt/data/Streamlit_Data.csv
# Updated: Route recommendation now uses 'optimised_route_preference' column order directly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Tourist Route & Insights", layout="wide")

# ----------------------------- Helpers ---------------------------------

def load_data(path: str = '/mnt/data/Streamlit_Data.csv') -> pd.DataFrame:
    """Loads dataset and normalises key column names for consistent access."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize possible alternative column names
    name_mappings = {
        'Destination': 'destination',
        'Place': 'destination',
        'place': 'destination',
        'Sequence': 'sequence',
        'Seq': 'sequence',
        'Crowd Density': 'crowd_density',
        'Traffic Level': 'traffic_level',
        'Festival Impact': 'festival_impact',
        'Average Cost': 'avg_cost',
        'Duration': 'duration'
    }
    df.rename(columns={k: v for k, v in name_mappings.items() if k in df.columns}, inplace=True)
    return df


def parse_sequence_cell(cell):
    """Converts text-based sequence entries into Python lists for easier processing."""
    if pd.isna(cell):
        return []
    s = str(cell)
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    for delim in ['->', '=>', ';', '|', ',']:
        if delim in s:
            return [p.strip() for p in s.split(delim) if p.strip()]
    return [s]


def build_adjacency_from_sequences(df):
    """Constructs adjacency mapping from the 'sequence' column for iterative destination selection."""
    adj = {}
    all_places = set()
    if 'sequence' not in df.columns:
        return adj, list(all_places)

    for seq in df['sequence'].dropna().unique():
        items = parse_sequence_cell(seq)
        for i, a in enumerate(items):
            all_places.add(a)
            if i + 1 < len(items):
                b = items[i + 1]
                adj.setdefault(a, set()).add(b)

    return {k: sorted(list(v)) for k, v in adj.items()}, sorted(list(all_places))

# ----------------------------- App UI -------------------------------------

st.markdown("""
<style>
.big-title{font-size:32px; font-weight:700; color:#2E86AB}
.small{color:#555}
.card{background:linear-gradient(135deg,#f8f9fb,#ffffff);padding:12px;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.06)}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Tourist Destinations — Interactive Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Visualise crowd, traffic, cost, duration & get optimized route suggestions</div>', unsafe_allow_html=True)
st.write('---')

# Load dataset
with st.spinner('Loading data...'):
    df = load_data()

# Sidebar: raw data preview
if st.sidebar.checkbox('Show raw data', value=False):
    st.dataframe(df)

# Build adjacency for iterative selection
adj, seq_places = build_adjacency_from_sequences(df)

# ---------------- Sidebar for itinerary selection ----------------
st.sidebar.header('Build an itinerary (iterative)')
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = []

# Determine next available destinations based on sequence adjacency
def next_options(current_itin):
    if adj:
        if not current_itin:
            return seq_places
        last = current_itin[-1]
        if last in adj and adj[last]:
            return adj[last]
        options = set()
        for seq in df['sequence'].dropna().unique():
            items =
