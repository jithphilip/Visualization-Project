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

def load_data(path: str = 'Streamlit_Data.csv') -> pd.DataFrame:
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
            items = parse_sequence_cell(seq)
            if last in items:
                idx = items.index(last)
                options.update(items[idx + 1:])
        return sorted(list(options))
    else:
        return sorted(df['sequence'].dropna().unique())

# Selection UI
cols = st.sidebar.columns([3, 1])
next_opts = next_options(st.session_state.itinerary)
if next_opts:
    choice = cols[0].selectbox('Choose next destination', ['-- none --'] + next_opts)
    if cols[1].button('Add') and choice != '-- none --':
        st.session_state.itinerary.append(choice)
        st.experimental_rerun()
else:
    st.sidebar.info('No next destinations available — try clearing itinerary.')

if st.sidebar.button('Clear itinerary'):
    st.session_state.itinerary = []
    st.experimental_rerun()

st.sidebar.markdown('**Current itinerary**')
if st.session_state.itinerary:
    for i, p in enumerate(st.session_state.itinerary, 1):
        st.sidebar.write(f"{i}. {p}")
else:
    st.sidebar.write('_No selections yet_')

# ---------------- Main display area ----------------

left, right = st.columns([2, 1])

with left:
    st.subheader('Itinerary summary')
    if not st.session_state.itinerary:
        st.info('Start by adding destinations from the sidebar to see metrics and route suggestions.')
    else:
        sel = df[df['sequence'].isin(st.session_state.itinerary)]

        # Compute averages for summary metrics
        def safe_mean(col):
            if col in sel.columns:
                try:
                    return sel[col].dropna().astype(float).mean()
                except:
                    return None
            return None

        crowd = safe_mean('crowd_density')
        traffic = safe_mean('traffic_level')
        fest = safe_mean('festival_impact')
        avg_cost = safe_mean('avg_cost')
        avg_duration = safe_mean('duration')

        # Metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Crowd (avg)', f"{crowd:.1f}" if crowd else 'N/A')
        c2.metric('Traffic (avg)', f"{traffic:.1f}" if traffic else 'N/A')
        c3.metric('Festival impact', f"{fest:.1f}" if fest else 'N/A')
        c4.metric('Avg cost', f"₹{avg_cost:.0f}" if avg_cost else 'N/A')
        c5.metric('Avg duration (hrs)', f"{avg_duration:.1f}" if avg_duration else 'N/A')

        st.markdown('---')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write('**Details for each selected destination**')
        st.dataframe(sel.reset_index(drop=True))
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- Route Recommendation ----------------
        st.subheader('Recommended Route (based on dataset preference)')

        # Instead of computing routes algorithmically, we now use the 'optimised_route_preference' column
        if 'optimised_route_preference' in df.columns:
            # Keep only rows for selected destinations
            route_df = df[df['sequence'].isin(st.session_state.itinerary)]
            # Sort by the provided preference order
            route_df = route_df.sort_values('optimised_route_preference')

            # Display the order
            for i, r in enumerate(route_df['destination'], 1):
                st.write(f"{i}. {r}")

            # # Show map if coordinates are available
            # if {'latitude', 'longitude'}.issubset(route_df.columns):
            #     map_df = route_df[['latitude', 'longitude', 'destination']]
            #     map_df = map_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            #     st.map(map_df)
            # else:
            #     st.info('No coordinate data available for map view.')
        else:
            st.warning("Column 'optimised_route_preference' not found in dataset — please add it to use this feature.")

# ---------------- Right panel for filtering ----------------
with right:
    st.subheader('Quick filters')
    filtered = df.copy()

    # Range sliders for numeric filters
    if 'crowd_density' in df.columns:
        minc, maxc = float(df['crowd_density'].min()), float(df['crowd_density'].max())
        crowd_range = st.slider('Crowd density range', min_value=minc, max_value=maxc, value=(minc, maxc))
        filtered = filtered[(filtered['crowd_density'] >= crowd_range[0]) & (filtered['crowd_density'] <= crowd_range[1])]

    if 'traffic_level' in df.columns:
        mint, maxt = float(df['traffic_level'].min()), float(df['traffic_level'].max())
        traffic_range = st.slider('Traffic level range', min_value=mint, max_value=maxt, value=(mint, maxt))
        filtered = filtered[(filtered['traffic_level'] >= traffic_range[0]) & (filtered['traffic_level'] <= traffic_range[1])]

    st.write(f'*{len(filtered)} destinations match filters*')
    st.dataframe(filtered[['destination']].drop_duplicates().reset_index(drop=True).head(10))

# ---------------- Visualisation Tabs ----------------
st.write('---')
tabs = st.tabs(['Univariate', 'Bivariate', 'Data Table & Export'])

# Univariate tab
with tabs[0]:
    st.header('Univariate Visualisations')
    col1, col2 = st.columns([2, 1])
    with col2:
        var = st.selectbox('Variable', options=df.columns.tolist())
        gtype = st.selectbox('Graph type', options=['Histogram', 'Box', 'Bar', 'Pie'])
        bins = st.slider('Bins (for histogram)', 5, 100, 20)
    with col1:
        if gtype == 'Histogram':
            fig = px.histogram(df, x=var, nbins=bins, title=f'Histogram of {var}')
            st.plotly_chart(fig, use_container_width=True)
        elif gtype == 'Box':
            fig = px.box(df, y=var, title=f'Box plot of {var}')
            st.plotly_chart(fig, use_container_width=True)
        elif gtype == 'Bar':
            vc = df[var].value_counts().nlargest(30)
            fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x': var, 'y': 'count'}, title=f'Bar of {var}')
            st.plotly_chart(fig, use_container_width=True)
        elif gtype == 'Pie':
            vc = df[var].value_counts().nlargest(10)
            fig = px.pie(values=vc.values, names=vc.index.astype(str), title=f'Pie of {var}')
            st.plotly_chart(fig, use_container_width=True)

# Bivariate tab
with tabs[1]:
    st.header('Bivariate Visualisations')
    xvar = st.selectbox('X variable', options=df.columns.tolist(), key='xvar')
    yvar = st.selectbox('Y variable', options=df.columns.tolist(), key='yvar')
    btype = st.selectbox('Plot type', options=['Scatter', 'Line', 'Heatmap', 'Box by group'])

    if btype == 'Scatter':
        fig = px.scatter(df, x=xvar, y=yvar, hover_data=df.columns, title=f'Scatter: {xvar} vs {yvar}')
        st.plotly_chart(fig, use_container_width=True)
    elif btype == 'Line':
        fig = px.line(df, x=xvar, y=yvar, title=f'Line: {xvar} vs {yvar}')
        st.plotly_chart(fig, use_container_width=True)
    elif btype == 'Heatmap':
        try:
            pivot = pd.crosstab(df[xvar], df[yvar])
            fig = px.imshow(pivot, title=f'Heatmap: {xvar} vs {yvar}')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.error('Cannot compute heatmap for these variables.')
    elif btype == 'Box by group':
        fig = px.box(df, x=xvar, y=yvar, title=f'Box of {yvar} grouped by {xvar}')
        st.plotly_chart(fig, use_container_width=True)

# Data Table tab
with tabs[2]:
    st.header('Data Table & Export')
    st.dataframe(filtered.reset_index(drop=True))
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered CSV', csv, 'filtered_data.csv', 'text/csv')

# Footer note
st.write('\n---\n')
st.markdown('**Note:** The route recommendation now strictly follows the order specified in the `optimised_route_preference` column.')
st.markdown('Feel free to modify this section to integrate your own route logic later.')


