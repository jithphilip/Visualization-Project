# streamlit_dashboard.py
# Streamlit dashboard for tourist-destination visualisation & route recommendations
# Reads data from Streamlit_Data.csv (expected next to this script)
# Updated to: 1) select sequence values iteratively, 2) aggregate metrics for matching rows,
# 3) use the 'optimised_route_preference' values from matching rows, and 4) iterative next-option narrowing

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter

st.set_page_config(page_title="Tourist Route & Insights", layout="wide")

# ----------------------------- Helpers ---------------------------------

def load_data(path: str = 'Streamlit_Data.csv') -> pd.DataFrame:
    """Loads dataset and normalises key column names for consistent access."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Common alternate names -> canonical names
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
        'Duration': 'duration',
        'Optimised Route Preference': 'optimised_route_preference'
    }
    df.rename(columns={k: v for k, v in name_mappings.items() if k in df.columns}, inplace=True)
    return df


def parse_sequence_cell(cell):
    """Converts text-based sequence entries into Python lists for easier processing.

    Accepts formats like:
      - comma separated: "A, B, C"
      - arrow separated: "A -> B -> C"
      - Python list string: "[A, B, C]"
      - semicolon/pipe separated

    Returns a list of stripped strings.
    """
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell]
    s = str(cell).strip()
    # strip surrounding brackets
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    # Common delimiters; try arrow-style first to avoid splitting on commas inside items
    for delim in ['->', '=>', '|', ';', ',']:
        if delim in s:
            parts = [p.strip() for p in s.split(delim) if p.strip()]
            return parts
    # fallback single token
    return [s]


def is_subsequence(small, big):
    """Return True if list `small` is a subsequence of list `big` preserving order (not necessarily contiguous).
    Example: [A,B] is subsequence of [X,A,Y,B,Z]."""
    if not small:
        return True
    bi = 0
    for item in big:
        if item == small[bi]:
            bi += 1
            if bi == len(small):
                return True
    return False


def rows_matching_itinerary(df, itinerary):
    """Return rows from df whose parsed sequence contains the `itinerary` as a subsequence.

    This is the key function: when the user builds an itinerary by selecting destinations
    iteratively, we keep only the dataset rows (sequence values) that still match the
    chosen itinerary in the same order. That filtered set is then used to compute
    aggregated metrics (crowd, traffic, festival, cost, duration) and to obtain
    the `optimised_route_preference` values corresponding to those sequences.
    """
    if 'sequence' not in df.columns:
        return df.iloc[0:0]  # empty frame with same columns

    mask_rows = []
    # Pre-parse sequences once for efficiency
    parsed_sequences = df['sequence'].fillna('').apply(parse_sequence_cell)
    for idx, seq_list in parsed_sequences.items():
        if is_subsequence(itinerary, seq_list):
            mask_rows.append(idx)
    return df.loc[mask_rows]


def next_options_from_matching(df, itinerary):
    """Given the current itinerary (list of selected destinations), return the list of
    possible next destinations by looking only at sequences that match the current itinerary.

    The logic:
      - Find all sequences (rows) where `itinerary` is a subsequence.
      - For each such sequence, find the elements that occur **after the last selected item** in that sequence.
      - Collect and return the sorted unique set of these next candidates.

    This ensures iterative narrowing: as itinerary grows, fewer sequences will match and
    available next options will shrink accordingly.
    """
    if 'sequence' not in df.columns:
        # no sequence info, return unique destination list
        return sorted(df['destination'].dropna().unique().tolist())

    # parse sequences
    parsed = df['sequence'].fillna('').apply(parse_sequence_cell)
    candidates = []
    # no selection yet: return all unique elements across sequences
    if not itinerary:
        all_items = set()
        for seq_list in parsed:
            all_items.update(seq_list)
        return sorted([i for i in all_items if i])

    last = itinerary[-1]
    for seq_list in parsed:
        if is_subsequence(itinerary, seq_list):
            # find positions of last in seq_list (could be multiple); collect all following items
            for pos, val in enumerate(seq_list):
                if val == last:
                    # add all items after this position
                    candidates.extend(seq_list[pos + 1 :])
    # unique, preserve sorted order
    return sorted(list(dict.fromkeys([c for c in candidates if c])))


# def aggregate_metrics_from_rows(rows):
#     """Given a DataFrame `rows` (matching sequences), compute the aggregated metrics required by the UI.

#     Returns a dict with keys: crowd_density, traffic_level, festival_impact, avg_cost, duration, optimised_route_preference_vals
#     - Numeric fields are averaged (mean) when multiple rows exist.
#     - optimised_route_preference_vals: returns unique list of values from matching rows (preserves order of appearance).
#     - If a field doesn't exist, value will be None.
#     """
#     out = {}
#     numeric_cols = ['crowd_density', 'traffic_level', 'festival_impact', 'avg_cost', 'duration']
#     for col in numeric_cols:
#         if col in rows.columns:
#             try:
#                 out[col] = float(rows[col].dropna().astype(float).mean())
#             except Exception:
#                 out[col] = None
#         else:
#             out[col] = None

#     # optimised route preference values: collect and dedupe preserving order
#     if 'optimised_route_preference' in rows.columns:
#         vals = rows['optimised_route_preference'].dropna().tolist()
#         # convert to strings for display
#         seen = []
#         for v in vals:
#             sv = str(v)
#             if sv not in seen:
#                 seen.append(sv)
#         out['optimised_route_preference_vals'] = seen
#         # if numeric, give a mode/median as helpful extra
#         try:
#             numeric_vals = [float(v) for v in vals]
#             out['optimised_route_preference_numeric_mode'] = float(pd.Series(numeric_vals).mode().iat[0]) if len(numeric_vals) else None
#         except Exception:
#             out['optimised_route_preference_numeric_mode'] = None
#     else:
#         out['optimised_route_preference_vals'] = []
#         out['optimised_route_preference_numeric_mode'] = None

#     return out

from statistics import mode, StatisticsError

def aggregate_metrics(df_subset):
    """Return mean for numeric fields and mode for categorical fields."""
    if df_subset.empty:
        return {}

    metrics = {}

    # Numeric columns (mean)
    for col in ['Total_Cost', 'Total_Duration']:
        if col in df_subset.columns:
            try:
                metrics[col] = round(df_subset[col].astype(float).mean(), 2)
            except Exception:
                metrics[col] = None

    # Categorical columns (mode)
    for col in ['Crowd_Density', 'Traffic_Level', 'Event_Impact', 'Weather']:
        if col in df_subset.columns:
            try:
                metrics[col] = mode(df_subset[col].dropna())
            except StatisticsError:
                metrics[col] = None  # if multiple modes or no data

    return metrics


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

# Build adjacency (not strictly necessary now since we derive next options from matching rows),
# but keep it for backwards compatibility or future use.
# (This function is used to compute next options in an alternative way.)
adj, seq_places = (None, [])
if 'sequence' in df.columns:
    # build simple adjacency (neighbors) for quick reference
    all_places = set()
    adj_temp = {}
    for seq in df['sequence'].dropna().unique():
        parsed = parse_sequence_cell(seq)
        for i, a in enumerate(parsed):
            all_places.add(a)
            if i + 1 < len(parsed):
                b = parsed[i + 1]
                adj_temp.setdefault(a, set()).add(b)
    adj = {k: sorted(list(v)) for k, v in adj_temp.items()}
    seq_places = sorted(list(all_places))

# ---------------- Sidebar for itinerary selection ----------------
st.sidebar.header('Build an itinerary (iterative)')
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = []

# Selection UI: show next options based on currently matching sequences
cols = st.sidebar.columns([3, 1])
next_opts = next_options_from_matching(df, st.session_state.itinerary)
if next_opts:
    choice = cols[0].selectbox('Choose next destination', ['-- none --'] + next_opts)
    if cols[1].button('Add') and choice != '-- none --':
        st.session_state.itinerary.append(choice)
        st.experimental_rerun()
else:
    st.sidebar.info('No next destinations available — try clearing itinerary or check your sequence data.')

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
        # Find rows where the sequence contains the selected itinerary as a subsequence
        sel = rows_matching_itinerary(df, st.session_state.itinerary)


        if not sel.empty:
            metrics = aggregate_metrics(sel)

            'Crowd_Density', 'Traffic_Level', 'Event_Impact', 'Weather'
            # ---- Metric cards ----
            c1, c2, c3, c4, c5, C6 = st.columns(6)
            c1.metric('Crowd (Most Frequent)', metrics.get('Crowd_Density', 'N/A'))
            c2.metric('Traffic (Most Frequent)', metrics.get('Traffic_Level', 'N/A'))
            c3.metric('Festival impact (Most Frequent)', metrics.get('Event_Impact', 'N/A'))
            c3.metric('Weather (Most Frequent)', metrics.get('Weather', 'N/A'))
            c5.metric('Avg cost', f"₹{metrics.get('Total_Cost', 'N/A')}")
            c6.metric('Avg duration (hrs)', metrics.get('Total_Duration', 'N/A'))
        
            st.markdown('---')
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(f"**{len(sel)} dataset rows match the selected itinerary.**")
            st.write('Details (first 50 rows):')
            st.dataframe(sel.head(50).reset_index(drop=True))
            st.markdown('</div>', unsafe_allow_html=True)

            # ---- Route recommendation ----
            st.subheader('Corresponding Optimised Route Preference')
            if 'Optimal_Route_Preference' in sel.columns:
                vals = sel['Optimal_Route_Preference'].dropna().unique().tolist()
                if vals:
                    for i, v in enumerate(vals, 1):
                        st.write(f"{i}. {v}")
                else:
                    st.info("No optimised route preference data available.")
            else:
                st.info("No 'optimised_route_preference' column found.")
        else:
            st.info('No matching records found for the selected itinerary.')



            

            # # ---------------- Route Recommendation ----------------
            # st.subheader('Corresponding Optimised Route Preference')
            # orp_vals = agg.get('optimised_route_preference_vals', [])
            # if orp_vals:
            #     st.write('Values from matching rows (preserving dataset order):')
            #     for i, v in enumerate(orp_vals, 1):
            #         st.write(f"{i}. {v}")
            #     if agg.get('optimised_route_preference_numeric_mode') is not None:
            #         st.write(f"Numeric mode of optimised_route_preference: {agg['optimised_route_preference_numeric_mode']}")
            # else:
            #     st.info("No 'optimised_route_preference' values found in the matching dataset rows.")

# ---------------- Right panel for filtering ----------------
with right:
    st.subheader('Quick filters')
    filtered = df.copy()

    # Range sliders for numeric filters
    if 'crowd_density' in df.columns:
        try:
            minc, maxc = float(df['crowd_density'].min()), float(df['crowd_density'].max())
            crowd_range = st.slider('Crowd density range', min_value=minc, max_value=maxc, value=(minc, maxc))
            filtered = filtered[(filtered['crowd_density'] >= crowd_range[0]) & (filtered['crowd_density'] <= crowd_range[1])]
        except Exception:
            pass

    if 'traffic_level' in df.columns:
        try:
            mint, maxt = float(df['traffic_level'].min()), float(df['traffic_level'].max())
            traffic_range = st.slider('Traffic level range', min_value=mint, max_value=maxt, value=(mint, maxt))
            filtered = filtered[(filtered['traffic_level'] >= traffic_range[0]) & (filtered['traffic_level'] <= traffic_range[1])]
        except Exception:
            pass

    st.write(f'*{len(filtered)} destinations match filters*')
    if 'destination' in filtered.columns:
        st.dataframe(filtered[['destination']].drop_duplicates().reset_index(drop=True).head(10))
    else:
        st.dataframe(filtered.head(10))

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
st.markdown('**Note:** The selection of next destinations is derived from sequences that still match the current itinerary; metrics are aggregated over those matching dataset rows.')
st.markdown('Modify `rows_matching_itinerary` and `next_options_from_matching` functions to change the matching rules or aggregation behavior.')





