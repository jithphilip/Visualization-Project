###############################################################################################################################


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statistics import mode

# ----------------------------------------------------------
# Page configuration
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
# Loading data
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
# Helper functions
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
# Tabs for Summary, Analysis and Itinerary planning
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Summary","Univariate Analysis", "Bivariate Analysis", "Itinerary Planner", "Route Finder"])

# ---------------------------------------------------------
# Tab 1
#----------------------------------------------------------
with tab1:
    st.markdown("""
    ### Project Summary

    This dashboard provides an exploratory analysis and interactive exploration of a dynamic tourism route dataset. 
    The goal is to support travellers and tourism planners in making informed decisions by analysing how route characteristics, 
    contextual factors, and user preferences influence overall travel satisfaction and route optimization.

    ---

    ## Dataset Overview

    The dataset consists of **1,345 travel routes** across **50 tourist destinations**, each representing a unique travel experience. 
    It contains **18 variables**, grouped into four major components:

    ### **1. Route Information**
    Describes the structural aspects of each travel route:
    - **Route ID** – Unique identifier for each route  
    - **Sequence** – Ordered list of destinations visited  
    - **Total Duration** – Time required to complete the route (minutes)  
    - **Total Cost** – Total expenditure for the route  

    ### **2. Dynamic Context Variables**
    Environmental and situational factors affecting travel experience:
    - **Weather** – Sunny, Rainy, Cloudy, Snowy  
    - **Traffic Level** – Low, Medium, High  
    - **Crowd Density** – Low, Medium, High  
    - **Event Impact** – Festival, Holiday, None  

    ### **3. User Information & Feedback**
    Captures demographic details and satisfaction:
    - **User ID**, **Age**, **Gender**, **Nationality**  
    - **Travel Companions** – Solo, Family, Friends, Group  
    - **Budget Category** – Low, Medium, High  
    - **Preferred Theme** – Adventure, Cultural, Shopping, etc.  
    - **Preferred Transport** – Car, Bus, Train, Taxi, Walk, Bike  
    - **Satisfaction Score** – Rating on a scale of 1–5  

    ### **4. System-Generated Recommendation**
    - **Optimal Route Preference** – A suggested reordered itinerary to maximize satisfaction while minimizing cost and duration  

    ---

    ## 📊 Dashboard Functionalities

    ### **1. Univariate Analysis**
    Visualizes individual variables using:
    - Bar charts
    - Histograms
    - Pie charts  
    Helps understand distributions, frequency patterns, and dominant categories.

    ### **2. Bivariate Analysis**
    Explores relationships between two variables:
    - **Categorical vs Categorical** → clustered bar charts  
    - **Categorical vs Numerical** → boxplots  
    - **Numerical vs Numerical** → scatter plots  
    Useful for identifying correlations—e.g.,  
    traffic vs satisfaction, weather vs duration, budget vs cost.

    ### **3. Route Search Utility**
    Allows users to:
    - Select a set of destinations  
    - Retrieve matching routes  
    Helps travellers filter routes based on their interests and available attractions.

    ### **4. Route Recommendation Module**
    Uses the **Optimal Route Preference** column to:
    - Suggest a reordered or improved route  
    - Compare chosen destinations with recommended order  
    Supports more efficient and experience-optimized travel planning.

    # ---

    ## Relevance to Traveller Problems

    These features collectively address common travel challenges such as:
    - Choosing routes that balance **time, cost, and satisfaction**
    - Understanding how **weather, traffic, and crowd levels** impact travel  
    - Identifying routes suited to specific **budget levels and travel preferences**  
    - Making informed decisions through **visual insights**  
    - Receiving **data-driven route recommendations** to improve trip planning  

    This dashboard thus serves as a practical tool for analysing real-world travel patterns 
    and enhancing route decisions using data-driven insights.
    """)



# ------------------------------------------------------------
# TAB 2: Univariate analysis
#-------------------------------------------------------------
with tab2:
    st.header("Univariate Analysis")
    st.write("Explore the distribution of individual variables.")

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

# ------------------------------------------------------------
# TAB 3: Bivariate analysis
# ------------------------------------------------------------
with tab3:
    st.header("Bivariate Analysis")
    st.write("Visualize relationships between two variables.")

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Select X variable", df.columns)
    with col2:
        y_var = st.selectbox("Select Y variable", df.columns)

    chart_type = st.selectbox(
        "Select chart type",
        ["Scatter Plot", "Box Plot", "Heatmap", "Grouped Bar Chart"]
    )

    numeric_cols = ['Total_Cost', 'Total_Duration', 'Satisfaction_Score']
    cat_cols = ['Crowd_Density', 'Traffic_Level', 'Event_Impact', 'Weather',
                'Travel_Companions', 'Preferred_Theme', 'Preferred_Transport']

    if chart_type == "Scatter Plot" and x_var in numeric_cols and y_var in numeric_cols:
        fig = px.scatter(df, x=x_var, y=y_var, hover_data=df.columns,
                         title=f"Scatter: {x_var} vs {y_var}")
        st.plotly_chart(fig, use_container_width=True)


    elif chart_type == "Box Plot" and (x_var in cat_cols and y_var in numeric_cols):
        fig = px.box(df, x=x_var, y=y_var, title=f"Box plot of {y_var} grouped by {x_var}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap" and (x_var in cat_cols and y_var in cat_cols):
        try:
            pivot = pd.crosstab(df[x_var], df[y_var])
            fig = px.imshow(pivot, title=f"Heatmap: {x_var} vs {y_var}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Cannot compute heatmap for these variables: {e}")

    elif chart_type == "Grouped Bar Chart" and (x_var in cat_cols and y_var in cat_cols):
        grouped_df = df.groupby([x_var, y_var]).size().reset_index(name="Count")
        fig = px.bar(
            grouped_df,
            x=x_var,
            y="Count",
            color=y_var,
            barmode="group",
            title=f"Grouped Bar Chart: {x_var} vs {y_var}"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Select suitable variable types for the chosen plot type.")

# ------------------------------------------------------------
# TAB 4: Itinerary Planner (with sidebar)
# ------------------------------------------------------------
with tab4:
    st.header("Itinerary Planner")

    st.sidebar.header("Build an itinerary [Select your options one by one]")

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

            # Filters matching rows and compute aggregated info
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
# Tab 5: Route Finder Tab
# ----------------------------------------------------------
with tab5:
    st.header("Find Routes Based on Your Preferences")

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

    # Generates filter widgets dynamically
    cols = st.columns(2)
    for i, col in enumerate(cat_filters.keys()):
        with cols[i % 2]:
            if col in route_df.columns:
                opts = sorted(route_df[col].dropna().unique().tolist())
                selected = st.multiselect(f"Select {col}", opts)
                if selected:
                    route_df = route_df[route_df[col].isin(selected)]
                    cat_filters[col] = selected

    # --- Shows results ---
    st.write("---")
    st.subheader("Matching Routes")

    if len(route_df) == 0:
        st.info("No routes found for the selected filters. Try widening your budget or removing some filters.")
    else:
        # Sorting routes by Total_Cost ascending
        route_df = route_df.sort_values(by="Total_Cost", ascending=True)
        st.dataframe(route_df[["sequence", "Total_Cost", "Optimal_Route_Preference"] + 
                               [c for c in cat_filters.keys() if c in route_df.columns]].reset_index(drop=True))

        # Optional route summary
        st.markdown(f"**{len(route_df)} routes found matching your preferences.**")
        avg_cost = route_df["Total_Cost"].mean() if "Total_Cost" in route_df.columns else None
        if avg_cost:
            st.metric("Average Cost of Matching Routes", f"₹{avg_cost:.0f}")



























