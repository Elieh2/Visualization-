import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set page config
st.set_page_config(
    page_title="Lebanon Education Resources Dashboard",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set default template
pio.templates.default = "plotly_white"

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .metric-container h3 {
        color: #2E8B57;
        margin-top: 0;
    }
    .metric-container h2 {
        color: #000000;
        margin: 0.5rem 0;
    }
    .metric-container p {
        color: #666666;
        margin-bottom: 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: black;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #2E8B57;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .insight-box h3 {
        color: #2E8B57;
        margin-top: 0;
    }
    .insight-box ul {
        margin-bottom: 0;
    }
    .insight-box li {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data function
@st.cache_data
def load_and_prepare_data():
    # Load the dataset
    df = pd.read_csv("766496d731ca34aa96a88c60f595617f_20240906_113458.csv")

    # Display available columns for debugging
    st.write("Available columns in the dataset:")
    st.write(df.columns.tolist())
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    st.write("Columns after stripping whitespace:")
    st.write(df.columns.tolist())

    # Data cleaning and preparation
    # Convert relevant columns to numeric, handling errors
    numeric_columns = [
        'Existence of educational resources - exists',
        'Type and size of educational resources - vocational institute',
        'Existence of educational resources - does not exist',
        'Nb of universities by type - Lebanese University branches',
        'Public school coverage index (number of schools per citizen)',
        'Type and size of educational resources - public schools',
        'Type and size of educational resources - universities',
        'Nb of universities by type - Private universities',
        'Type and size of educational resources - private schools'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            st.warning(f"Column '{col}' not found in dataset")

    # Check what refArea column contains and extract district
    if 'refArea' in df.columns:
        st.write("Sample refArea values:")
        st.write(df['refArea'].head(10))
        
        # Extract district from refArea - let's see what the pattern is
        df['District'] = df['refArea'].apply(lambda x: str(x).split('/')[-1].replace('_', ' ').title())
        
        st.write("Extracted District values:")
        st.write(df['District'].unique())
    else:
        st.error("refArea column not found in dataset")
        # Create a dummy District column for testing
        df['District'] = 'Unknown'

    # Clean district names
    district_mapping = {
        'Mount Lebanon District': 'Mount Lebanon',
        'South District': 'South',
        'Akkar District': 'Akkar',
        'North District': 'North',
        'Nabatieh District': 'Nabatieh',
        'Beqaa District': 'Beqaa',
        'Baalbek-Hermel District': 'Baalbek-Hermel',
        'Mount Lebanon': 'Mount Lebanon',
        'South': 'South',
        'Akkar': 'Akkar',
        'North': 'North',
        'Nabatieh': 'Nabatieh',
        'Beqaa': 'Beqaa',
        'Baalbek-Hermel': 'Baalbek-Hermel'
    }

    df['District'] = df['District'].replace(district_mapping)

    # FEATURE ENGINEERING SECTION
    # 1. Calculate total educational resources per town
    resource_cols = [
        'Type and size of educational resources - public schools',
        'Type and size of educational resources - private schools',
        'Type and size of educational resources - universities',
        'Type and size of educational resources - vocational institute'
    ]
    
    # Initialize missing columns with zeros
    for col in resource_cols:
        if col not in df.columns:
            df[col] = 0

    df['Total Educational Resources'] = (
        df['Type and size of educational resources - public schools'] +
        df['Type and size of educational resources - private schools'] +
        df['Type and size of educational resources - universities'] +
        df['Type and size of educational resources - vocational institute']
    )

    # 2. Calculate resource density (resources per estimated population)
    if 'Public school coverage index (number of schools per citizen)' in df.columns:
        df['Estimated Population'] = np.where(
            df['Public school coverage index (number of schools per citizen)'] > 0,
            df['Type and size of educational resources - public schools'] / 
            df['Public school coverage index (number of schools per citizen)'],
            df['Total Educational Resources'] * 1000  # Default estimation
        )
    else:
        df['Estimated Population'] = df['Total Educational Resources'] * 1000

    # Calculate resource density
    df['Resource Density (per 1000 people)'] = np.where(
        df['Estimated Population'] > 0,
        (df['Total Educational Resources'] / df['Estimated Population']) * 1000,
        0
    )

    # 3. Create a resource type dominance feature
    df['Dominant Resource Type'] = df[resource_cols].idxmax(axis=1)
    df['Dominant Resource Type'] = df['Dominant Resource Type'].apply(
        lambda x: 'Public Schools' if 'public' in str(x).lower() else
                  'Private Schools' if 'private' in str(x).lower() else
                  'Universities' if 'universities' in str(x).lower() else
                  'Vocational Institutes'
    )

    # 4. Create binary flags for different resource types
    df['Has Public Schools'] = df['Type and size of educational resources - public schools'] > 0
    df['Has Private Schools'] = df['Type and size of educational resources - private schools'] > 0
    df['Has Universities'] = df['Type and size of educational resources - universities'] > 0
    df['Has Vocational Institutes'] = df['Type and size of educational resources - vocational institute'] > 0

    # 5. Create a composite education index (weighted average of resources)
    # Normalize each resource type
    for col in resource_cols:
        max_val = df[col].max()
        if max_val > 0:  # Avoid division by zero
            df[f'{col}_normalized'] = df[col] / max_val
        else:
            df[f'{col}_normalized'] = 0

    # Create weighted education index
    df['Education Index'] = (
        df['Type and size of educational resources - public schools_normalized'] * 0.4 +
        df['Type and size of educational resources - private schools_normalized'] * 0.3 +
        df['Type and size of educational resources - universities_normalized'] * 0.2 +
        df['Type and size of educational resources - vocational institute_normalized'] * 0.1
    )

    # 6. Create resource diversity metric
    df['Resource Diversity'] = df[resource_cols].std(axis=1) / (df[resource_cols].mean(axis=1) + 0.001)

    # 7. Create urban/rural classification based on resource density
    try:
        df['Area Type'] = pd.qcut(df['Resource Density (per 1000 people)'], 
                                  q=3, 
                                  labels=['Rural', 'Suburban', 'Urban'])
    except:
        df['Area Type'] = 'Rural'
    
    return df

# Load data
df = load_and_prepare_data()

# Check if District column exists before proceeding
if 'District' not in df.columns:
    st.error("District column was not created successfully. Please check your data.")
    st.stop()

# Create summary statistics by district
try:
    district_summary = df.groupby('District').agg({
        'Existence of educational resources - exists': 'sum',
        'Type and size of educational resources - public schools': 'sum',
        'Type and size of educational resources - private schools': 'sum',
        'Type and size of educational resources - universities': 'sum',
        'Type and size of educational resources - vocational institute': 'sum',
        'Town': 'count',
        'Total Educational Resources': 'sum',
        'Estimated Population': 'sum',
        'Education Index': 'mean',
        'Resource Density (per 1000 people)': 'mean'
    }).reset_index()

    district_summary.columns = [
        'District', 'Towns with Resources', 'Public Schools', 
        'Private Schools', 'Universities', 'Vocational Institutes', 'Total Towns',
        'Total Resources', 'Total Population', 'Avg Education Index', 'Avg Resource Density'
    ]

    district_summary['Resource Coverage (%)'] = (district_summary['Towns with Resources'] / district_summary['Total Towns'] * 100).round(1)
except Exception as e:
    st.error(f"Error creating district summary: {e}")
    # Create empty summary
    district_summary = pd.DataFrame(columns=[
        'District', 'Towns with Resources', 'Public Schools', 
        'Private Schools', 'Universities', 'Vocational Institutes', 'Total Towns',
        'Total Resources', 'Resource Coverage (%)'
    ])

# Dashboard Header
st.markdown('<h1 class="main-header">🏫 Lebanon Education Resources Dashboard</h1>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.markdown("## 🔍 Dashboard Filters")
st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)

# District filter
selected_districts = st.sidebar.multiselect(
    "Select Districts:",
    options=df['District'].unique(),
    default=df['District'].unique(),
    help="Choose which districts to display in the analysis"
)

# Resource threshold filter
min_resources = st.sidebar.slider(
    "Minimum Total Resources:",
    min_value=int(df['Total Educational Resources'].min()),
    max_value=int(df['Total Educational Resources'].max()),
    value=int(df['Total Educational Resources'].min()),
    help="Filter towns by minimum number of educational resources"
)

# School type dominance filter
dominant_options = ['All'] + list(df['Dominant Resource Type'].unique())
selected_dominant = st.sidebar.selectbox(
    "Dominant Resource Type:",
    options=dominant_options,
    help="Filter by the dominant type of educational resource"
)

# Area type filter
area_options = ['All'] + list(df['Area Type'].unique())
selected_area = st.sidebar.selectbox(
    "Area Type:",
    options=area_options,
    help="Filter by urban, suburban, or rural classification"
)

# Education index range filter
min_edu, max_edu = st.sidebar.slider(
    "Education Index Range:",
    min_value=float(df['Education Index'].min()),
    max_value=float(df['Education Index'].max()),
    value=(float(df['Education Index'].min()), float(df['Education Index'].max())),
    help="Filter towns by education index range"
)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Apply filters
filtered_df = df[df['District'].isin(selected_districts)]
filtered_df = filtered_df[filtered_df['Total Educational Resources'] >= min_resources]
filtered_df = filtered_df[filtered_df['Education Index'].between(min_edu, max_edu)]

if selected_dominant != 'All':
    filtered_df = filtered_df[filtered_df['Dominant Resource Type'] == selected_dominant]

if selected_area != 'All':
    filtered_df = filtered_df[filtered_df['Area Type'] == selected_area]

# Create summary by district for filtered data
if len(filtered_df) > 0:
    try:
        filtered_district_summary = filtered_df.groupby('District').agg({
            'Existence of educational resources - exists': 'sum',
            'Type and size of educational resources - public schools': 'sum',
            'Type and size of educational resources - private schools': 'sum',
            'Type and size of educational resources - universities': 'sum',
            'Type and size of educational resources - vocational institute': 'sum',
            'Town': 'count',
            'Total Educational Resources': 'sum',
            'Estimated Population': 'sum',
            'Education Index': 'mean',
            'Resource Density (per 1000 people)': 'mean'
        }).reset_index()

        filtered_district_summary.columns = [
            'District', 'Towns with Resources', 'Public Schools', 
            'Private Schools', 'Universities', 'Vocational Institutes', 'Total Towns',
            'Total Resources', 'Total Population', 'Avg Education Index', 'Avg Resource Density'
        ]

        filtered_district_summary['Resource Coverage (%)'] = (filtered_district_summary['Towns with Resources'] / filtered_district_summary['Total Towns'] * 100).round(1)
    except Exception as e:
        st.error(f"Error creating filtered district summary: {e}")
        filtered_district_summary = pd.DataFrame()
else:
    filtered_district_summary = pd.DataFrame()

# Key Metrics Row - Updated with 3 metrics instead of 4
st.markdown("## 📊 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    total_towns = len(filtered_df)
    st.markdown(f"""
    <div class="metric-container">
        <h3>🏘️ Total Towns</h3>
        <h2>{total_towns}</h2>
        <p>Towns in selected areas</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_coverage = filtered_district_summary['Resource Coverage (%)'].mean() if len(filtered_district_summary) > 0 else 0
    st.markdown(f"""
    <div class="metric-container">
        <h3>📈 Avg Coverage</h3>
        <h2>{avg_coverage:.1f}%</h2>
        <p>Resource coverage rate</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if len(filtered_district_summary) > 0:
        top_district = filtered_district_summary.loc[filtered_district_summary['Resource Coverage (%)'].idxmax(), 'District']
        top_value = filtered_district_summary.loc[filtered_district_summary['Resource Coverage (%)'].idxmax(), 'Resource Coverage (%)']
    else:
        top_district = 'N/A'
        top_value = 0
        
    st.markdown(f"""
    <div class="metric-container">
        <h3>🏆 Top Performer</h3>
        <h2>{top_district}</h2>
        <p>{top_value:.1f}% coverage</p>
    </div>
    """, unsafe_allow_html=True)

# Main visualizations
st.markdown("---")

# Visualization 1: Bar Chart - Educational Resource Coverage by District (EXACT COPY from your original code)
if len(filtered_district_summary) > 0:
    st.markdown("## 📊 Educational Resource Coverage by District")
    
    # EXACT same chart as your original code - Visualization 1
    fig1 = px.bar(
        filtered_district_summary, 
        x='District', 
        y='Resource Coverage (%)',
        title='Educational Resource Coverage by District',
        color='District',
        text='Resource Coverage (%)',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig1.update_layout(
        xaxis_title="District",
        yaxis_title="Percentage of Towns with Educational Resources",
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )

    fig1.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)

    # Insights for Chart 1
    max_coverage = filtered_district_summary['Resource Coverage (%)'].max()
    min_coverage = filtered_district_summary['Resource Coverage (%)'].min()
    max_district = filtered_district_summary.loc[filtered_district_summary['Resource Coverage (%)'].idxmax(), 'District']
    min_district = filtered_district_summary.loc[filtered_district_summary['Resource Coverage (%)'].idxmin(), 'District']
    
    st.markdown(f"""
    <div class="insight-box">
        <h3>🔍 Key Insights - Resource Coverage</h3>
        <ul>
            <li><strong>Coverage Range:</strong> Resource coverage ranges from {min_coverage}% in {min_district} to {max_coverage}% in {max_district}</li>
            <li><strong>Performance Gap:</strong> There's a {max_coverage - min_coverage:.1f} percentage point difference between the highest and lowest performing districts</li>
            <li><strong>Average Coverage:</strong> The average resource coverage across selected districts is {filtered_district_summary['Resource Coverage (%)'].mean():.1f}%</li>
            <li><strong>Policy Focus:</strong> {min_district} may need targeted investment to improve educational resource coverage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Visualization 2: Scatter Plot - EXACT REPLICA of your original code (3rd visualization)
if len(filtered_district_summary) > 0:
    st.markdown("## 🏫 Public vs Private Schools Distribution")

    # EXACT same chart as your original code - Visualization 3
    fig2 = px.scatter(
        filtered_district_summary,
        x='Public Schools',
        y='Private Schools',
        size='Total Towns',
        color='District',
        title='Public vs Private Schools Distribution by District',
        hover_name='District',
        size_max=60,
        labels={
            'Public Schools': 'Number of Public Schools',
            'Private Schools': 'Number of Private Schools'
        }
    )

    fig2.update_layout(
        xaxis_title="Number of Public Schools",
        yaxis_title="Number of Private Schools"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Dynamic insights based on filtered data
    total_public = filtered_district_summary['Public Schools'].sum() if len(filtered_district_summary) > 0 else 0
    total_private = filtered_district_summary['Private Schools'].sum() if len(filtered_district_summary) > 0 else 0
    public_private_ratio = f"{total_public}/{total_private}" if total_private > 0 else f"{total_public}/0"
    
    if len(filtered_district_summary) > 0:
        public_leader = filtered_district_summary.loc[filtered_district_summary['Public Schools'].idxmax(), 'District']
        private_leader = filtered_district_summary.loc[filtered_district_summary['Private Schools'].idxmax(), 'District']
    else:
        public_leader = 'N/A'
        private_leader = 'N/A'

    st.markdown(f"""
    <div class="insight-box">
        <h3>🔍 Dynamic Insights - Public vs Private Distribution</h3>
        <ul>
            <li><strong>Selected Areas Ratio:</strong> {public_private_ratio} (Public/Private schools)</li>
            <li><strong>Public School Concentration:</strong> {public_leader} leads in public schools</li>
            <li><strong>Private School Hub:</strong> {private_leader} has most private schools</li>
            <li><strong>Bubble Size Meaning:</strong> Larger bubbles = more towns in that district</li>
            <li><strong>Correlation:</strong> {'Positive' if filtered_district_summary['Public Schools'].corr(filtered_district_summary['Private Schools']) > 0 else 'Negative'} correlation between public and private schools</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("No data available for the selected filters. Please adjust your filter settings.")
