# streamlit_dashboard.py
# Streamlit dashboard for tourist-destination visualisation & route recommendations
# Reads data from /mnt/data/Streamlit_Data.csv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from math import radians, cos, sin, asin, sqrt

st.set_page_config(page_title="Tourist Route & Insights", layout="wide")

# ----------------------------- Helpers ---------------------------------

def load_data(path: str = '/mnt/data/Streamlit_Data.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # ensure some expected columns exist for downstream features
    if 'destination' not in df.columns:
        # attempt to detect a column that looks like destination
        for guess in ['Destination', 'place', 'Place', 'name']:
            if guess in df.columns:
                df = df.rename(columns={guess: 'destination'})
                break
    # sequence could be stored under many names
    seq_cols = [c for c in df.columns if 'seq' in c.lower()] 
    if seq_cols:
        df = df.rename(columns={seq_cols[0]: 'sequence'})
    # crowd/traffic etc fallback names
    for col in ['crowd_density','traffic_level','festival_impact','avg_cost','duration']:
        if col not in df.columns:
            for guess in [col.title(), col.replace('_',' ').title(), col.replace('_',' ').lower()]:
                if guess in df.columns:
                    df = df.rename(columns={guess: col})
                    break
    return df


def parse_sequence_cell(cell):
    """Try to parse a sequence cell into a list of destinations.
    Accepts comma-separated, -> separated, semicolon, or Python list strings."""
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return list(cell)
    s = str(cell)
    # remove surrounding brackets
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    # delimiters
    for delim in ['->', '=>', ';', '|']:
        if delim in s:
            parts = [p.strip() for p in s.split(delim) if p.strip()]
            return parts
    # fallback comma
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return parts


def build_adjacency_from_sequences(df):
    """Return adjacency dictionary and all unique destinations found in sequences."""
    adj = {}
    all_places = set()
    if 'sequence' not in df.columns:
        return adj, list(all_places)
    for seq in df['sequence'].dropna().unique():
        items = parse_sequence_cell(seq)
        for i, a in enumerate(items):
            all_places.add(a)
            if i+1 < len(items):
                b = items[i+1]
                adj.setdefault(a, set()).add(b)
    return {k: sorted(list(v)) for k, v in adj.items()}, sorted(list(all_places))


def haversine(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km


def greedy_route(coords):
    # coords: list of (name, lat, lon)
    if not coords:
        return []
    unvisited = coords.copy()
    route = [unvisited.pop(0)]  # start from first selected location
    while unvisited:
        last = route[-1]
        dists = [haversine(last[1], last[2], u[1], u[2]) for u in unvisited]
        idx = int(np.argmin(dists))
        route.append(unvisited.pop(idx))
    return route

# ----------------------------- App -------------------------------------

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

# load data
with st.spinner('Loading data...'):
    df = load_data()

# preview
if st.sidebar.checkbox('Show raw data', value=False):
    st.dataframe(df)

# build adjacency
adj, seq_places = build_adjacency_from_sequences(df)

# Sidebar controls - iterative destination selector
st.sidebar.header('Build an itinerary (iterative)')
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = []

# function to compute next options based on current itinerary
def next_options(current_itin):
    # If we have sequences and adjacency, options are neighbors of last selected across sequences
    if adj:
        if not current_itin:
            # starting options: all destinations that appear in sequences
            return seq_places
        last = current_itin[-1]
        # options are adjacency of last; if no adjacency found, allow any place that appears after last in any sequence
        if last in adj and adj[last]:
            return adj[last]
        # fallback: any place that co-occurs in sequences after last
        options = set()
        for seq in df['sequence'].dropna().unique():
            items = parse_sequence_cell(seq)
            if last in items:
                idx = items.index(last)
                options.update(items[idx+1:])
        return sorted(list(options))
    else:
        # if no sequence info, allow all unique destinations
        return sorted(df['destination'].dropna().unique())

# interactive add/remove
cols = st.sidebar.columns([3,1])
next_opts = next_options(st.session_state.itinerary)
if not next_opts:
    st.sidebar.info('No next destinations available from current selection — try clearing itinerary or check sequence data')
else:
    choice = cols[0].selectbox('Choose next destination', ['-- none --'] + next_opts)
    if cols[1].button('Add') and choice and choice != '-- none --':
        st.session_state.itinerary.append(choice)
        st.experimental_rerun()

if st.sidebar.button('Clear itinerary'):
    st.session_state.itinerary = []
    st.experimental_rerun()

st.sidebar.markdown('**Current itinerary**')
if st.session_state.itinerary:
    for i, place in enumerate(st.session_state.itinerary, 1):
        st.sidebar.write(f"{i}. {place}")
else:
    st.sidebar.write('_No selections yet_')

# Main layout
left, right = st.columns([2,1])

with left:
    st.subheader('Itinerary summary')
    if not st.session_state.itinerary:
        st.info('Start by adding destinations from the sidebar to see metrics and route recommendations')
    else:
        sel = df[df['destination'].isin(st.session_state.itinerary)]
        # compute aggregate metrics
        def safe_mean(col):
            if col in sel.columns:
                return sel[col].dropna().astype(float).mean()
            return None
        crowd = safe_mean('crowd_density')
        traffic = safe_mean('traffic_level')
        fest = safe_mean('festival_impact')
        avg_cost = safe_mean('avg_cost')
        avg_duration = safe_mean('duration')

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Crowd (avg)', f"{crowd:.1f}" if crowd is not None else 'N/A')
        c2.metric('Traffic (avg)', f"{traffic:.1f}" if traffic is not None else 'N/A')
        c3.metric('Festival impact', f"{fest:.1f}" if fest is not None else 'N/A')
        c4.metric('Avg cost', f"₹{avg_cost:.0f}" if avg_cost is not None else 'N/A')
        c5.metric('Avg duration (hrs)', f"{avg_duration:.1f}" if avg_duration is not None else 'N/A')

        st.markdown('---')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write('**Details for each selected destination**')
        st.dataframe(sel.reset_index(drop=True))
        st.markdown('</div>', unsafe_allow_html=True)

        # Route optimization
        st.subheader('Recommended route')
        # if coordinates exist
        if {'latitude','longitude'}.issubset(set(df.columns)):
            coords = []
            for place in st.session_state.itinerary:
                row = df[df['destination'] == place].iloc[0]
                coords.append((place, float(row['latitude']), float(row['longitude'])))
            route = greedy_route(coords)
            route_names = [r[0] for r in route]
            st.write('Route (greedy nearest-neighbour):')
            for i, r in enumerate(route_names,1):
                st.write(f"{i}. {r}")
            # show map
            map_df = pd.DataFrame([(r[1], r[2], r[0]) for r in route], columns=['lat','lon','name'])
            st.map(map_df.rename(columns={'lat':'lat','lon':'lon'}))
        else:
            # fallback: try to use sequence order present in sequences
            # find a common sequence that contains whole itinerary in order
            found_seq = None
            if 'sequence' in df.columns:
                for seq in df['sequence'].dropna().unique():
                    items = parse_sequence_cell(seq)
                    # check if itinerary is subsequence of items
                    it = st.session_state.itinerary
                    idx = 0
                    for it_item in it:
                        if it_item in items:
                            pos = items.index(it_item)
                            if pos >= idx:
                                idx = pos + 1
                                continue
                            else:
                                break
                        else:
                            break
                    else:
                        # passed
                        found_seq = items
                        break
            if found_seq:
                st.write('Optimized order based on an existing sequence in the dataset:')
                # intersect found_seq with itinerary preserving order
                optimized = [x for x in found_seq if x in st.session_state.itinerary]
                for i, p in enumerate(optimized,1):
                    st.write(f"{i}. {p}")
            else:
                # final fallback: sort by duration ascending
                if 'duration' in df.columns:
                    sorted_sel = sel.sort_values('duration')
                    st.write('Optimized by shortest duration first:')
                    for i, p in enumerate(sorted_sel['destination'],1):
                        st.write(f"{i}. {p}")
                else:
                    st.write('No coordinates or sequence heuristics available — displaying selected list')
                    for i, p in enumerate(st.session_state.itinerary,1):
                        st.write(f"{i}. {p}")

with right:
    st.subheader('Quick filters')
    # allow filtering by crowd/traffic ranges if present
    if 'crowd_density' in df.columns:
        minc, maxc = float(df['crowd_density'].min()), float(df['crowd_density'].max())
        crowd_range = st.slider('Crowd density range', min_value=minc, max_value=maxc, value=(minc, maxc))
    else:
        crowd_range = None
    if 'traffic_level' in df.columns:
        mint, maxt = float(df['traffic_level'].min()), float(df['traffic_level'].max())
        traffic_range = st.slider('Traffic level range', min_value=mint, max_value=maxt, value=(mint, maxt))
    else:
        traffic_range = None

    # apply quick filter to dataframe sample
    filtered = df.copy()
    if crowd_range and 'crowd_density' in df.columns:
        filtered = filtered[(filtered['crowd_density'] >= crowd_range[0]) & (filtered['crowd_density'] <= crowd_range[1])]
    if traffic_range and 'traffic_level' in df.columns:
        filtered = filtered[(filtered['traffic_level'] >= traffic_range[0]) & (filtered['traffic_level'] <= traffic_range[1])]
    st.write(f'*{len(filtered)} destinations match filters*')
    st.dataframe(filtered[['destination']].drop_duplicates().reset_index(drop=True).head(10))

# Tabs for plots
st.write('---')
tabs = st.tabs(['Univariate', 'Bivariate', 'Data Table & Export'])

with tabs[0]:
    st.header('Univariate Visualisations')
    col1, col2 = st.columns([2,1])
    with col2:
        var = st.selectbox('Variable', options=df.columns.tolist())
        gtype = st.selectbox('Graph type', options=['Histogram', 'Box', 'Bar', 'Pie'])
        bins = st.slider('Bins (histogram)', 5, 100, 20)
    with col1:
        if gtype == 'Histogram':
            try:
                fig = px.histogram(df, x=var, nbins=bins, title=f'Histogram of {var}')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(e)
        elif gtype == 'Box':
            fig = px.box(df, y=var, title=f'Box plot of {var}')
            st.plotly_chart(fig, use_container_width=True)
        elif gtype == 'Bar':
            vc = df[var].value_counts().nlargest(30)
            fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x':var,'y':'count'}, title=f'Bar of {var}')
            st.plotly_chart(fig, use_container_width=True)
        elif gtype == 'Pie':
            vc = df[var].value_counts().nlargest(10)
            fig = px.pie(values=vc.values, names=vc.index.astype(str), title=f'Pie of {var}')
            st.plotly_chart(fig, use_container_width=True)

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
        except Exception as e:
            st.error('Cannot compute heatmap for selected variables')
    elif btype == 'Box by group':
        fig = px.box(df, x=xvar, y=yvar, title=f'Box of {yvar} grouped by {xvar}')
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header('Data Table & Export')
    st.write('Preview filtered dataset and export to CSV')
    st.dataframe(filtered.reset_index(drop=True))
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered CSV', csv, 'filtered_data.csv', 'text/csv')

st.write('\n---\n')
st.markdown('**Notes & further ideas:**')
st.markdown('- The iterative selector in the sidebar uses `sequence` column (if present) to offer only valid next stops based on observed sequences in the dataset.')
st.markdown('- Route optimization: uses coordinates if available (greedy nearest neighbour). If coordinates are not available, the app tries to use sequence order or sorts by duration.')
st.markdown('- You can extend the app to use a full TSP solver (OR-Tools) if higher-quality routes are needed.')
st.markdown('- Add user-adjustable weighting (e.g., prioritize low crowd over short duration) to produce custom route optimised for user preferences.')

st.markdown('<div style="font-size:12px;color:#666">Built with ♥ — edit the file to tune styling and visual choices.</div>', unsafe_allow_html=True)
