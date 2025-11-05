# # streamlit_dashboard.py
# # Streamlit dashboard for tourist-destination visualisation & route recommendations
# # Reads data from Streamlit_Data.csv (expected next to this script)
# # Updated to: 1) select sequence values iteratively, 2) aggregate metrics for matching rows,
# # 3) use the 'optimised_route_preference' values from matching rows, and 4) iterative next-option narrowing

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from collections import Counter

# st.set_page_config(page_title="Tourist Route & Insights", layout="wide")

# # ----------------------------- Helpers ---------------------------------

# def load_data(path: str = 'Streamlit_Data.csv') -> pd.DataFrame:
#     """Loads dataset and normalises key column names for consistent access."""
#     df = pd.read_csv(path)
#     df.columns = [c.strip() for c in df.columns]

#     # Common alternate names -> canonical names
#     name_mappings = {
#         'Destination': 'destination',
#         'Place': 'destination',
#         'place': 'destination',
#         'Sequence': 'sequence',
#         'Seq': 'sequence',
#         'Crowd Density': 'crowd_density',
#         'Traffic Level': 'traffic_level',
#         'Festival Impact': 'festival_impact',
#         'Average Cost': 'avg_cost',
#         'Duration': 'duration',
#         'Optimised Route Preference': 'optimised_route_preference'
#     }
#     df.rename(columns={k: v for k, v in name_mappings.items() if k in df.columns}, inplace=True)
#     return df


# def parse_sequence_cell(cell):
#     """Converts text-based sequence entries into Python lists for easier processing.

#     Accepts formats like:
#       - comma separated: "A, B, C"
#       - arrow separated: "A -> B -> C"
#       - Python list string: "[A, B, C]"
#       - semicolon/pipe separated

#     Returns a list of stripped strings.
#     """
#     if pd.isna(cell):
#         return []
#     if isinstance(cell, (list, tuple)):
#         return [str(x).strip() for x in cell]
#     s = str(cell).strip()
#     # strip surrounding brackets
#     if s.startswith('[') and s.endswith(']'):
#         s = s[1:-1]
#     # Common delimiters; try arrow-style first to avoid splitting on commas inside items
#     for delim in ['->', '=>', '|', ';', ',']:
#         if delim in s:
#             parts = [p.strip() for p in s.split(delim) if p.strip()]
#             return parts
#     # fallback single token
#     return [s]


# def is_subsequence(small, big):
#     """Return True if list `small` is a subsequence of list `big` preserving order (not necessarily contiguous).
#     Example: [A,B] is subsequence of [X,A,Y,B,Z]."""
#     if not small:
#         return True
#     bi = 0
#     for item in big:
#         if item == small[bi]:
#             bi += 1
#             if bi == len(small):
#                 return True
#     return False


# def rows_matching_itinerary(df, itinerary):
#     """Return rows from df whose parsed sequence contains the `itinerary` as a subsequence.

#     This is the key function: when the user builds an itinerary by selecting destinations
#     iteratively, we keep only the dataset rows (sequence values) that still match the
#     chosen itinerary in the same order. That filtered set is then used to compute
#     aggregated metrics (crowd, traffic, festival, cost, duration) and to obtain
#     the `optimised_route_preference` values corresponding to those sequences.
#     """
#     if 'sequence' not in df.columns:
#         return df.iloc[0:0]  # empty frame with same columns

#     mask_rows = []
#     # Pre-parse sequences once for efficiency
#     parsed_sequences = df['sequence'].fillna('').apply(parse_sequence_cell)
#     for idx, seq_list in parsed_sequences.items():
#         if is_subsequence(itinerary, seq_list):
#             mask_rows.append(idx)
#     return df.loc[mask_rows]


# def next_options_from_matching(df, itinerary):
#     """Given the current itinerary (list of selected destinations), return the list of
#     possible next destinations by looking only at sequences that match the current itinerary.

#     The logic:
#       - Find all sequences (rows) where `itinerary` is a subsequence.
#       - For each such sequence, find the elements that occur **after the last selected item** in that sequence.
#       - Collect and return the sorted unique set of these next candidates.

#     This ensures iterative narrowing: as itinerary grows, fewer sequences will match and
#     available next options will shrink accordingly.
#     """
#     if 'sequence' not in df.columns:
#         # no sequence info, return unique destination list
#         return sorted(df['destination'].dropna().unique().tolist())

#     # parse sequences
#     parsed = df['sequence'].fillna('').apply(parse_sequence_cell)
#     candidates = []
#     # no selection yet: return all unique elements across sequences
#     if not itinerary:
#         all_items = set()
#         for seq_list in parsed:
#             all_items.update(seq_list)
#         return sorted([i for i in all_items if i])

#     last = itinerary[-1]
#     for seq_list in parsed:
#         if is_subsequence(itinerary, seq_list):
#             # find positions of last in seq_list (could be multiple); collect all following items
#             for pos, val in enumerate(seq_list):
#                 if val == last:
#                     # add all items after this position
#                     candidates.extend(seq_list[pos + 1 :])
#     # unique, preserve sorted order
#     return sorted(list(dict.fromkeys([c for c in candidates if c])))


# # def aggregate_metrics_from_rows(rows):
# #     """Given a DataFrame `rows` (matching sequences), compute the aggregated metrics required by the UI.

# #     Returns a dict with keys: crowd_density, traffic_level, festival_impact, avg_cost, duration, optimised_route_preference_vals
# #     - Numeric fields are averaged (mean) when multiple rows exist.
# #     - optimised_route_preference_vals: returns unique list of values from matching rows (preserves order of appearance).
# #     - If a field doesn't exist, value will be None.
# #     """
# #     out = {}
# #     numeric_cols = ['crowd_density', 'traffic_level', 'festival_impact', 'avg_cost', 'duration']
# #     for col in numeric_cols:
# #         if col in rows.columns:
# #             try:
# #                 out[col] = float(rows[col].dropna().astype(float).mean())
# #             except Exception:
# #                 out[col] = None
# #         else:
# #             out[col] = None

# #     # optimised route preference values: collect and dedupe preserving order
# #     if 'optimised_route_preference' in rows.columns:
# #         vals = rows['optimised_route_preference'].dropna().tolist()
# #         # convert to strings for display
# #         seen = []
# #         for v in vals:
# #             sv = str(v)
# #             if sv not in seen:
# #                 seen.append(sv)
# #         out['optimised_route_preference_vals'] = seen
# #         # if numeric, give a mode/median as helpful extra
# #         try:
# #             numeric_vals = [float(v) for v in vals]
# #             out['optimised_route_preference_numeric_mode'] = float(pd.Series(numeric_vals).mode().iat[0]) if len(numeric_vals) else None
# #         except Exception:
# #             out['optimised_route_preference_numeric_mode'] = None
# #     else:
# #         out['optimised_route_preference_vals'] = []
# #         out['optimised_route_preference_numeric_mode'] = None

# #     return out

# from statistics import mode, StatisticsError

# def aggregate_metrics(df_subset):
#     """Return mean for numeric fields and mode for categorical fields."""
#     if df_subset.empty:
#         return {}

#     metrics = {}

#     # Numeric columns (mean)
#     for col in ['Total_Cost', 'Total_Duration']:
#         if col in df_subset.columns:
#             try:
#                 metrics[col] = round(df_subset[col].astype(float).mean(), 2)
#             except Exception:
#                 metrics[col] = None

#     # Categorical columns (mode)
#     for col in ['Crowd_Density', 'Traffic_Level', 'Event_Impact', 'Weather']:
#         if col in df_subset.columns:
#             try:
#                 metrics[col] = mode(df_subset[col].dropna())
#             except StatisticsError:
#                 metrics[col] = None  # if multiple modes or no data

#     return metrics


# # ----------------------------- App UI -------------------------------------

# st.markdown("""
# <style>
# .big-title{font-size:32px; font-weight:700; color:#2E86AB}
# .small{color:#555}
# .card{background:linear-gradient(135deg,#f8f9fb,#ffffff);padding:12px;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.06)}
# </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="big-title">Tourist Destinations — Interactive Dashboard</div>', unsafe_allow_html=True)
# st.markdown('<div class="small">Visualise crowd, traffic, cost, duration & get optimized route suggestions</div>', unsafe_allow_html=True)
# st.write('---')

# # Load dataset
# with st.spinner('Loading data...'):
#     df = load_data()

# # Sidebar: raw data preview
# if st.sidebar.checkbox('Show raw data', value=False):
#     st.dataframe(df)

# # Build adjacency (not strictly necessary now since we derive next options from matching rows),
# # but keep it for backwards compatibility or future use.
# # (This function is used to compute next options in an alternative way.)
# adj, seq_places = (None, [])
# if 'sequence' in df.columns:
#     # build simple adjacency (neighbors) for quick reference
#     all_places = set()
#     adj_temp = {}
#     for seq in df['sequence'].dropna().unique():
#         parsed = parse_sequence_cell(seq)
#         for i, a in enumerate(parsed):
#             all_places.add(a)
#             if i + 1 < len(parsed):
#                 b = parsed[i + 1]
#                 adj_temp.setdefault(a, set()).add(b)
#     adj = {k: sorted(list(v)) for k, v in adj_temp.items()}
#     seq_places = sorted(list(all_places))

# # ---------------- Sidebar for itinerary selection ----------------
# st.sidebar.header('Build an itinerary (iterative)')
# if 'itinerary' not in st.session_state:
#     st.session_state.itinerary = []

# # Selection UI: show next options based on currently matching sequences
# cols = st.sidebar.columns([3, 1])
# next_opts = next_options_from_matching(df, st.session_state.itinerary)
# if next_opts:
#     choice = cols[0].selectbox('Choose next destination', ['-- none --'] + next_opts)
#     if cols[1].button('Add') and choice != '-- none --':
#         st.session_state.itinerary.append(choice)
#         st.experimental_rerun()
# else:
#     st.sidebar.info('No next destinations available — try clearing itinerary or check your sequence data.')

# if st.sidebar.button('Clear itinerary'):
#     st.session_state.itinerary = []
#     st.experimental_rerun()

# st.sidebar.markdown('**Current itinerary**')
# if st.session_state.itinerary:
#     for i, p in enumerate(st.session_state.itinerary, 1):
#         st.sidebar.write(f"{i}. {p}")
# else:
#     st.sidebar.write('_No selections yet_')

# # ---------------- Main display area ----------------
# left, right = st.columns([2, 1])

# with left:
#     st.subheader('Itinerary summary')
#     st.markdown("""<div class="itinerary-summary">""", unsafe_allow_html=True)
#     if not st.session_state.itinerary:
#         st.info('Start by adding destinations from the sidebar to see metrics and route suggestions.')
#     else:
#         # Find rows where the sequence contains the selected itinerary as a subsequence
#         sel = rows_matching_itinerary(df, st.session_state.itinerary)


#         if not sel.empty:
#             metrics = aggregate_metrics(sel)

#             'Crowd_Density', 'Traffic_Level', 'Event_Impact', 'Weather'
#             # ---- Metric cards ----
#             c1, c2, c3, c4, c5, c6 = st.columns(6)
#             c1.metric('Crowd (Most Frequent)', metrics.get('Crowd_Density', 'N/A'))
#             c2.metric('Traffic (Most Frequent)', metrics.get('Traffic_Level', 'N/A'))
#             c3.metric('Festival impact (Most Frequent)', metrics.get('Event_Impact', 'N/A'))
#             c3.metric('Weather (Most Frequent)', metrics.get('Weather', 'N/A'))
#             c5.metric('Avg cost', f"₹{metrics.get('Total_Cost', 'N/A')}")
#             c6.metric('Avg duration (hrs)', metrics.get('Total_Duration', 'N/A'))
        
#             # st.markdown('---')
#             st.markdown("""<style>.itinerary-summary div[data-testid="stMetricValue"] { font-size:0.50rem !important; }.itinerary-summary div[data-testid="stMetricLabel"] { font-size:0.50rem !important; }</style>""", unsafe_allow_html=True)
#             st.write(f"**{len(sel)} dataset rows match the selected itinerary.**")
#             st.write('Details for selected Itenary [From Data]:')
#             st.dataframe(sel.head(50).reset_index(drop=True))
#             st.markdown("</div>", unsafe_allow_html=True)
#             # ---- Route recommendation ----
#             st.subheader('Corresponding Optimised Route Preference')
#             if 'Optimal_Route_Preference' in sel.columns:
#                 vals = sel['Optimal_Route_Preference'].dropna().unique().tolist()
#                 if vals:
#                     for i, v in enumerate(vals, 1):
#                         st.write(f"{i}. {v}")
#                 else:
#                     st.info("No optimised route preference data available.")
#             else:
#                 st.info("No 'optimised_route_preference' column found.")
#         else:
#             st.info('No matching records found for the selected itinerary.')



            

#             # # ---------------- Route Recommendation ----------------
#             # st.subheader('Corresponding Optimised Route Preference')
#             # orp_vals = agg.get('optimised_route_preference_vals', [])
#             # if orp_vals:
#             #     st.write('Values from matching rows (preserving dataset order):')
#             #     for i, v in enumerate(orp_vals, 1):
#             #         st.write(f"{i}. {v}")
#             #     if agg.get('optimised_route_preference_numeric_mode') is not None:
#             #         st.write(f"Numeric mode of optimised_route_preference: {agg['optimised_route_preference_numeric_mode']}")
#             # else:
#             #     st.info("No 'optimised_route_preference' values found in the matching dataset rows.")

# # ---------------- Right panel for filtering ----------------
# with right:
#     st.subheader('Quick filters')
#     filtered = df.copy()

#     # Range sliders for numeric filters
#     if 'crowd_density' in df.columns:
#         try:
#             minc, maxc = float(df['crowd_density'].min()), float(df['crowd_density'].max())
#             crowd_range = st.slider('Crowd density range', min_value=minc, max_value=maxc, value=(minc, maxc))
#             filtered = filtered[(filtered['crowd_density'] >= crowd_range[0]) & (filtered['crowd_density'] <= crowd_range[1])]
#         except Exception:
#             pass

#     if 'traffic_level' in df.columns:
#         try:
#             mint, maxt = float(df['traffic_level'].min()), float(df['traffic_level'].max())
#             traffic_range = st.slider('Traffic level range', min_value=mint, max_value=maxt, value=(mint, maxt))
#             filtered = filtered[(filtered['traffic_level'] >= traffic_range[0]) & (filtered['traffic_level'] <= traffic_range[1])]
#         except Exception:
#             pass

#     st.write(f'*{len(filtered)} destinations match filters*')
#     if 'destination' in filtered.columns:
#         st.dataframe(filtered[['destination']].drop_duplicates().reset_index(drop=True).head(10))
#     else:
#         st.dataframe(filtered.head(10))

# # ---------------- Visualisation Tabs ----------------
# st.write('---')
# tabs = st.tabs(['Univariate', 'Bivariate', 'Data Table & Export'])

# # Univariate tab
# with tabs[0]:
#     st.header('Univariate Visualisations')
#     col1, col2 = st.columns([2, 1])
#     with col2:
#         var = st.selectbox('Variable', options=df.columns.tolist())
#         gtype = st.selectbox('Graph type', options=['Histogram', 'Box', 'Bar', 'Pie'])
#         bins = st.slider('Bins (for histogram)', 5, 100, 20)
#     with col1:
#         if gtype == 'Histogram':
#             fig = px.histogram(df, x=var, nbins=bins, title=f'Histogram of {var}')
#             st.plotly_chart(fig, use_container_width=True)
#         elif gtype == 'Box':
#             fig = px.box(df, y=var, title=f'Box plot of {var}')
#             st.plotly_chart(fig, use_container_width=True)
#         elif gtype == 'Bar':
#             vc = df[var].value_counts().nlargest(30)
#             fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x': var, 'y': 'count'}, title=f'Bar of {var}')
#             st.plotly_chart(fig, use_container_width=True)
#         elif gtype == 'Pie':
#             vc = df[var].value_counts().nlargest(10)
#             fig = px.pie(values=vc.values, names=vc.index.astype(str), title=f'Pie of {var}')
#             st.plotly_chart(fig, use_container_width=True)

# # Bivariate tab
# with tabs[1]:
#     st.header('Bivariate Visualisations')
#     xvar = st.selectbox('X variable', options=df.columns.tolist(), key='xvar')
#     yvar = st.selectbox('Y variable', options=df.columns.tolist(), key='yvar')
#     btype = st.selectbox('Plot type', options=['Scatter', 'Line', 'Heatmap', 'Box by group'])

#     if btype == 'Scatter':
#         fig = px.scatter(df, x=xvar, y=yvar, hover_data=df.columns, title=f'Scatter: {xvar} vs {yvar}')
#         st.plotly_chart(fig, use_container_width=True)
#     elif btype == 'Line':
#         fig = px.line(df, x=xvar, y=yvar, title=f'Line: {xvar} vs {yvar}')
#         st.plotly_chart(fig, use_container_width=True)
#     elif btype == 'Heatmap':
#         try:
#             pivot = pd.crosstab(df[xvar], df[yvar])
#             fig = px.imshow(pivot, title=f'Heatmap: {xvar} vs {yvar}')
#             st.plotly_chart(fig, use_container_width=True)
#         except:
#             st.error('Cannot compute heatmap for these variables.')
#     elif btype == 'Box by group':
#         fig = px.box(df, x=xvar, y=yvar, title=f'Box of {yvar} grouped by {xvar}')
#         st.plotly_chart(fig, use_container_width=True)

# # Data Table tab
# with tabs[2]:
#     st.header('Data Table & Export')
#     st.dataframe(filtered.reset_index(drop=True))
#     csv = filtered.to_csv(index=False).encode('utf-8')
#     st.download_button('Download filtered CSV', csv, 'filtered_data.csv', 'text/csv')

# # Footer note
# st.write('\n---\n')
# st.markdown('**Note:** The selection of next destinations is derived from sequences that still match the current itinerary; metrics are aggregated over those matching dataset rows.')
# st.markdown('Modify `rows_matching_itinerary` and `next_options_from_matching` functions to change the matching rules or aggregation behavior.')

###############################################################################################################################


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statistics import mode

# ----------------------------------------------------------
# 🎨 Page configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Tourism Dashboard", layout="wide")
st.markdown("""
<style>
/* General aesthetic */
h1, h2, h3 {
    color: #E0E0E0;
}
section[data-testid="stSidebar"] {
    background-color: #1E1E1E;
}
div[data-testid="stMetricValue"] {
    font-size: 20px;  /* reduce text size for metrics */
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 📂 Load data
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Streamlit_Data.csv")
    df["sequence"] = df["Sequence"].apply(
        lambda x: [i.strip() for i in x.strip("[]").split(",")] if isinstance(x, str) else []
    )
    return df

df = load_data()

# ----------------------------------------------------------
# 🧩 Helper functions
# ----------------------------------------------------------
def get_next_destinations(selected_dests, df):
    """
    Given a list of already selected destinations, returns
    the next possible destinations based on 'sequence' column.
    """
    filtered = df[df["sequence"].apply(lambda seq: all(d in seq for d in selected_dests))]
    next_dests = set()
    for seq in filtered["sequence"]:
        for i, d in enumerate(seq):
            if i > 0 and seq[i-1] == selected_dests[-1]:
                next_dests.add(d)
    return sorted(next_dests)

def get_summary_stats(df_subset):
    """
    Returns mean for numeric columns and mode for categorical ones.
    """
    numeric_cols = ['Total_Cost', 'Total_Duration', 'Satisfaction_Score']
    cat_cols = ['Crowd_Density', 'Traffic_Level', 'Event_Impact', 'Weather', 'Travel_Companions', 'Preferred_Theme', 'Preferred_Transport']

    summary = {}
    for col in numeric_cols:
        summary[col] = round(df_subset[col].mean(), 2)
    for col in cat_cols:
        try:
            summary[col] = mode(df_subset[col].dropna().tolist())
        except:
            summary[col] = df_subset[col].dropna().mode()[0] if not df_subset[col].dropna().empty else None
    return summary

# ----------------------------------------------------------
# 📊 Tabs for analysis and itinerary planning
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Itinerary Planner", "Route Finder"])

# ----------------------------------------------------------
# 🟢 TAB 1: Univariate analysis
# ----------------------------------------------------------
with tab1:
    st.header("Univariate Analysis")
    st.write("Explore the distribution of individual variables.")

    #df.select_dtypes(include=['number']).columns.tolist()
    #df.select_dtypes(exclude=['number']).columns.tolist()
    numeric_cols = ['Total_Cost', 'Total_Duration', 'Satisfaction_Score']
    cat_cols = ['Crowd_Density', 'Traffic_Level', 'Event_Impact', 'Weather', 'Travel_Companions', 'Preferred_Theme', 'Preferred_Transport']

    var = st.selectbox("Select variable for analysis", df.columns)
    chart_type = st.selectbox("Select chart type", ["Histogram", "Boxplot", "Bar Chart", "Pie Chart"])

    if var:
        if chart_type == "Histogram" and var in numeric_cols:
            fig = px.histogram(df, x=var, nbins=30, title=f"Histogram of {var}")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Boxplot" and var in numeric_cols:
            fig = px.box(df, y=var, title=f"Boxplot of {var}")
            st.plotly_chart(fig, use_container_width=True)

        # elif chart_type == "Bar Chart" and var in cat_cols:
        #     fig = px.bar(df[var].value_counts().reset_index(),
        #                  x='index', y=var,
        #                  labels={'index': var, var: 'Count'},
        #                  title=f"Bar chart of {var}")
        #     st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Bar Chart" and var in cat_cols:
            bar_df = df[var].value_counts().reset_index()
            bar_df.columns = [var, "Count"]  # rename columns properly
            fig = px.bar(bar_df, x=var, y="Count",
                         labels={var: var, "Count": "Count"},
                         title=f"Bar chart of {var}")
            st.plotly_chart(fig, use_container_width=True)


        elif chart_type == "Pie Chart" and var in cat_cols:
            fig = px.pie(df, names=var, title=f"Pie chart of {var}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selected chart type is not suitable for this variable type.")

# ----------------------------------------------------------
# 🟡 TAB 2: Bivariate analysis
# ----------------------------------------------------------
with tab2:
    st.header("Bivariate Analysis")
    st.write("Visualize relationships between two variables.")

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Select X variable", df.columns)
    with col2:
        y_var = st.selectbox("Select Y variable", df.columns)

    chart_type = st.selectbox("Select chart type", ["Scatter", "Line", "Box", "Bar"])

    if x_var and y_var:
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_var, y=y_var, title=f"Scatter plot of {x_var} vs {y_var}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_var, y=y_var, title=f"Line plot of {x_var} vs {y_var}")
        elif chart_type == "Box":
            fig = px.box(df, x=x_var, y=y_var, title=f"Box plot of {x_var} vs {y_var}")
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_var, y=y_var, title=f"Bar chart of {x_var} vs {y_var}")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# 🔵 TAB 3: Itinerary Planner (with sidebar)
# ----------------------------------------------------------
with tab3:
    st.header("Itinerary Planner")

    st.sidebar.header("Build an itinerary (iterative)")

    # Iterative destination selection
    selected_dests = []
    all_sequences = [seq for seq in df['sequence'] if isinstance(seq, list)]

    first_destinations = sorted({dest for seq in all_sequences for dest in seq})
    first_selection = st.sidebar.selectbox("Select your first destination", ["None"] + first_destinations)

    if first_selection != "None":
        selected_dests.append(first_selection)

        while True:
            next_dests = get_next_destinations(selected_dests, df)
            if not next_dests:
                break
            next_sel = st.sidebar.selectbox(f"Next destination after {selected_dests[-1]}", ["None"] + next_dests, key=f"sel_{len(selected_dests)}")
            if next_sel == "None":
                break
            selected_dests.append(next_sel)

        if selected_dests:
            st.subheader("Your Itinerary Summary")

            # Filter matching rows and compute aggregated info
            filtered_df = df[df["sequence"].apply(lambda seq: all(d in seq for d in selected_dests))]
            if not filtered_df.empty:
                summary = get_summary_stats(filtered_df)

                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                col1.metric("Crowd Intensity", summary.get("Crowd_Density", "N/A"))
                col2.metric("Traffic Level", summary.get("Traffic_Level", "N/A"))
                col3.metric("Event Impact", summary.get("Event_Impact", "N/A"))
                col4.metric("Weather", summary.get("Weather", "N/A"))
                col5.metric("Travel Companions", summary.get("Travel_Companions", "N/A"))
                col6.metric("Preferred Theme", summary.get("Preferred_Theme", "N/A"))
                col7.metric("Preferred Transport", summary.get("Preferred_Transport", "N/A"))
                
                col8, col9, col10 = st.columns(3)
                col8.metric("Average Cost ($)", summary.get("Total_Cost", "N/A"))
                col9.metric("Average Duration (hrs)", summary.get("Total_Duration", "N/A"))
                col10.metric("Average Rating (1-5)", summary.get("Satisfaction_Score", "N/A"))
                # ----------------------------------------------------------
                # 🚗 Route Recommendation
                # ----------------------------------------------------------
                if "Optimal_Route_Preference" in df.columns:
                    route_values = filtered_df["Optimal_Route_Preference"].dropna().unique().tolist()
                    if len(route_values) == 0:
                        st.markdown("### 🚗 Recommended Route: Not available for the selected destinations.")
                    else:
                        # If multiple routes exist, pick the most frequent one (mode)
                        try:
                            recommended_route = mode(filtered_df["Optimal_Route_Preference"].dropna().tolist())
                        except:
                            recommended_route = route_values[0]  # fallback to first available
                        st.markdown(f"### 🚗 Recommended Route Order: {recommended_route}")

            else:
                st.info("No matching itinerary found for the selected destinations.")


# ----------------------------------------------------------
# Tab 4:🧭 Route Finder Tab
# ----------------------------------------------------------
with tab4:
    st.header("🧭 Find Routes Based on Your Preferences")

    # Create a working copy of df
    route_df = df.copy()

    # --- Budget filter ---
    if "Total_Cost" in route_df.columns:
        min_cost, max_cost = int(route_df["Total_Cost"].min()), int(route_df["Total_Cost"].max())
        budget = st.slider("💰 Select your total budget range ($)", 
                           min_value=min_cost, max_value=max_cost, value=(min_cost, max_cost))
        route_df = route_df[(route_df["Total_Cost"] >= budget[0]) & (route_df["Total_Cost"] <= budget[1])]
    else:
        st.warning("⚠️ 'Total_Cost' column not found in dataset.")

    st.markdown("### Optional Filters")

    # --- Optional categorical filters ---
    cat_filters = {
        "Crowd_Density": None,
        "Traffic_Level": None,
        "Event_Impact": None,
        "Weather": None,
        "Travel_Companions": None,
        "Preferred_Theme": None,
        "Preferred_Transport": None
    }

    # Generate filter widgets dynamically
    cols = st.columns(2)
    for i, col in enumerate(cat_filters.keys()):
        with cols[i % 2]:
            if col in route_df.columns:
                opts = sorted(route_df[col].dropna().unique().tolist())
                selected = st.multiselect(f"Select {col}", opts)
                if selected:
                    route_df = route_df[route_df[col].isin(selected)]
                    cat_filters[col] = selected

    # --- Show results ---
    st.write("---")
    st.subheader("🔍 Matching Routes")

    if len(route_df) == 0:
        st.info("No routes found for the selected filters. Try widening your budget or removing some filters.")
    else:
        # Sort routes by Total_Cost ascending
        route_df = route_df.sort_values(by="Total_Cost", ascending=True)
        st.dataframe(route_df[["sequence", "Total_Cost", "Optimal_Route_Preference"] + 
                               [c for c in cat_filters.keys() if c in route_df.columns]].reset_index(drop=True))

        # Optional route summary
        st.markdown(f"**{len(route_df)} routes found matching your preferences.**")
        avg_cost = route_df["Total_Cost"].mean() if "Total_Cost" in route_df.columns else None
        if avg_cost:
            st.metric("Average Cost of Matching Routes", f"₹{avg_cost:.0f}")

        # # Download filtered routes
        # csv = route_df.to_csv(index=False).encode("utf-8")
        # st.download_button("⬇️ Download Matching Routes", csv, "filtered_routes.csv", "text/csv")


















