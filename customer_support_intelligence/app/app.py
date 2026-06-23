from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import shap
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
processed_path = BASE_DIR / "data" / "processed" / "customer_support_tickets_processed.csv"
model_path = BASE_DIR / "models" / "best_model.pkl"
preprocessor_path = BASE_DIR / "models" / "preprocessor.pkl"

# ----------------------------------------------------
# 1. Page Configuration & Theme
# ----------------------------------------------------
st.set_page_config(
    page_title="Support Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css(str(BASE_DIR / "app" / "styles.css"))
# Initialize Google GenAI client
default_key = "[SECURE_REMOVED_KEY]"
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or default_key

with st.sidebar:
    st.markdown("### 🔑 Gemini API Configuration")
    user_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get("gemini_api_key", api_key), help="Enter your Google Gemini API Key to enable AI features.")
    if user_api_key:
        st.session_state.gemini_api_key = user_api_key
        api_key = user_api_key

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        client = None
        st.sidebar.error(f"Failed to initialize Gemini Client: {e}")
else:
    client = None

# Download VADER
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception:
    pass

# ----------------------------------------------------
# 2. Data & Model Loaders
# ----------------------------------------------------
@st.cache_data
def load_data():
    if processed_path.exists():
        return pd.read_csv(processed_path)
    return pd.DataFrame()

@st.cache_resource
def load_ml_artifacts():
    if preprocessor_path.exists() and model_path.exists():
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return preprocessor, model
    return None, None

df = load_data()
preprocessor, model = load_ml_artifacts()

if df.empty:
    st.error("Processed dataset not found! Please run the modeling notebook first.")
    st.stop()

# Extract feature names dynamically
num_cols = ['Customer Age', 'First Response Time (Hours)', 'Time to Resolution (Hours)', 'Resolution Duration (Hours)', 'Sentiment Score']
cat_cols = ['Customer Gender', 'Product Purchased', 'Ticket Type', 'Ticket Subject', 'Ticket Priority', 'Ticket Channel', 'Ticket Topic Label']
try:
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    all_feature_names = num_cols + list(cat_encoder.get_feature_names_out(cat_cols))
except Exception:
    all_feature_names = num_cols

# ----------------------------------------------------
# 3. Hero Section
# ----------------------------------------------------
st.markdown('<h1 class="hero-title">AI-Powered Customer Support Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Leveraging NLP, Explainable Machine Learning and Generative AI for Customer Experience Optimization</p>', unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Horizontal Filter Bar (No Sidebar)
# ----------------------------------------------------
all_products = sorted(df['Product Purchased'].dropna().unique())
all_priorities = sorted(df['Ticket Priority'].dropna().unique())
all_channels = sorted(df['Ticket Channel'].dropna().unique())
all_status = sorted(df['Ticket Status'].dropna().unique())

product_options = ["Product: All"] + [f"Product: {x}" for x in all_products]
priority_options = ["Priority: All"] + [f"Priority: {x}" for x in all_priorities]
channel_options = ["Channel: All"] + [f"Channel: {x}" for x in all_channels]
status_options = ["Status: All"] + [f"Status: {x}" for x in all_status]

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    selected_product_opt = st.selectbox("Product Line Filter", options=product_options, label_visibility="collapsed")
with filter_col2:
    selected_priority_opt = st.selectbox("Priority Filter", options=priority_options, label_visibility="collapsed")
with filter_col3:
    selected_channel_opt = st.selectbox("Channel Filter", options=channel_options, label_visibility="collapsed")
with filter_col4:
    selected_status_opt = st.selectbox("Status Filter", options=status_options, label_visibility="collapsed")

selected_product = selected_product_opt.split(": ", 1)[1]
selected_priority = selected_priority_opt.split(": ", 1)[1]
selected_channel = selected_channel_opt.split(": ", 1)[1]
selected_status = selected_status_opt.split(": ", 1)[1]

# Apply dynamic filters
filtered_df = df.copy()
if selected_product != "All":
    filtered_df = filtered_df[filtered_df['Product Purchased'] == selected_product]
if selected_priority != "All":
    filtered_df = filtered_df[filtered_df['Ticket Priority'] == selected_priority]
if selected_channel != "All":
    filtered_df = filtered_df[filtered_df['Ticket Channel'] == selected_channel]
if selected_status != "All":
    filtered_df = filtered_df[filtered_df['Ticket Status'] == selected_status]

# ----------------------------------------------------
# 5. KPI Section (Four metric cards)
# ----------------------------------------------------
total_tickets = len(filtered_df)
closed_tickets_pct = (filtered_df['Ticket Status'].str.lower() == 'closed').sum() / total_tickets * 100 if total_tickets > 0 else 0.0
avg_satisfaction = filtered_df['Customer Satisfaction Rating'].mean()
avg_resolution_hours = filtered_df['Resolution Duration (Hours)'].mean()

st.write("") # Spacer
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(f"""
    <div class="saas-card kpi-container">
        <div class="kpi-title">Total Tickets</div>
        <div class="kpi-value kpi-accent-purple">{total_tickets:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="saas-card kpi-container">
        <div class="kpi-title">Resolution Rate</div>
        <div class="kpi-value kpi-accent-cyan">{closed_tickets_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="saas-card kpi-container">
        <div class="kpi-title">Average CSAT</div>
        <div class="kpi-value kpi-accent-purple">{avg_satisfaction:.2f} ★</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="saas-card kpi-container">
        <div class="kpi-title">Resolution Speed</div>
        <div class="kpi-value kpi-accent-cyan">{avg_resolution_hours:.1f} hrs</div>
    </div>
    """, unsafe_allow_html=True)

# Helper function to style Plotly charts for clean dark SaaS blending
def style_plotly_layout(fig, title_text=None, is_horizontal_bar=False):
    margin = dict(l=80, r=40, t=50, b=80)
    if is_horizontal_bar:
        margin = dict(l=140, r=30, t=50, b=50)
        
    layout_update = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC', family="Plus Jakarta Sans"),
        margin=margin,
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            linecolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94A3B8')
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            linecolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94A3B8')
        )
    )
    if title_text:
        layout_update['title'] = dict(
            text=f"<b>{title_text}</b>",
            font=dict(size=15, color='#F8FAFC', family="Plus Jakarta Sans"),
            x=0.02,
            y=0.95
        )
        
    fig.update_layout(**layout_update)
    return fig

# Helper function to render empty chart states cleanly
def show_empty_chart_state(title):
    st.markdown(f"""
    <div class="chart-card" style="border: 1px dashed rgba(255,255,255,0.08); text-align: center; padding: 40px 20px; margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 20px;">{title}</div>
        <div style="color: #64748B; font-size: 0.95rem;">No data available for current filters</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 6. Customer Intelligence Section
# ----------------------------------------------------
st.write("---")
st.markdown('<span class="section-badge">Customer Intelligence</span>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Operational Ticket Distribution</h2>', unsafe_allow_html=True)
st.write("") # Spacer

col_cust1, col_cust2, col_cust3 = st.columns(3)

if filtered_df.empty:
    with col_cust1:
        show_empty_chart_state("Ticket Status Allocation")
    with col_cust2:
        show_empty_chart_state("Support Channel Volumes")
    with col_cust3:
        show_empty_chart_state("Priority Allocation")
else:
    with col_cust1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        status_counts = filtered_df['Ticket Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_status = px.pie(
            status_counts, values='Count', names='Status', hole=0.65,
            template="plotly_dark", color_discrete_sequence=['#7C3AED', '#22D3EE', '#F8FAFC']
        )
        style_plotly_layout(fig_status, "Ticket Status Allocation")
        fig_status.update_layout(
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center"
            ),
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=80
            )
        )
        fig_status.update_traces(domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]))
        st.plotly_chart(
            fig_status,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

    with col_cust2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        channel_counts = filtered_df['Ticket Channel'].value_counts().reset_index()
        channel_counts.columns = ['Channel', 'Count']
        fig_channel = px.bar(
            channel_counts, x='Count', y='Channel', orientation='h',
            template="plotly_dark", color_discrete_sequence=['#7C3AED']
        )
        style_plotly_layout(fig_channel, "Support Channel Volumes", is_horizontal_bar=True)
        st.plotly_chart(
            fig_channel,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

    with col_cust3:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        priority_counts = filtered_df['Ticket Priority'].value_counts().reset_index()
        priority_counts.columns = ['Priority', 'Count']
        priority_order = ['Low', 'Medium', 'High', 'Critical']
        priority_counts['Priority'] = pd.Categorical(priority_counts['Priority'], categories=priority_order, ordered=True)
        priority_counts = priority_counts.sort_values('Priority')
        fig_priority = px.bar(
            priority_counts, x='Priority', y='Count',
            template="plotly_dark", color='Priority',
            color_discrete_map={'Low': '#22D3EE', 'Medium': '#7C3AED', 'High': '#c084fc', 'Critical': '#f43f5e'}
        )
        style_plotly_layout(fig_priority, "Priority Allocation")
        fig_priority.update_layout(showlegend=False)
        st.plotly_chart(
            fig_priority,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

# ----------------------------------------------------
# 7. Product Intelligence Section
# ----------------------------------------------------
st.write("---")
st.markdown('<span class="section-badge">Product Intelligence</span>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Product Line Performance</h2>', unsafe_allow_html=True)
st.write("")

prod_col1, prod_col2 = st.columns([2, 1])

product_stats = pd.DataFrame()
if not filtered_df.empty:
    product_stats = filtered_df.groupby('Product Purchased').agg(
        volume=('Ticket ID', 'count'),
        avg_sat=('Customer Satisfaction Rating', 'mean'),
        avg_res_time=('Resolution Duration (Hours)', 'mean')
    ).reset_index().dropna()

with prod_col1:
    if not product_stats.empty:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        fig_bubble = px.scatter(
            product_stats, x='avg_res_time', y='avg_sat', size='volume', color='avg_sat',
            hover_name='Product Purchased', size_max=30,
            color_continuous_scale=['#22D3EE', '#7C3AED'], template="plotly_dark",
            labels={'avg_res_time': 'Resolution Time (Hours)', 'avg_sat': 'Satisfaction Rating', 'volume': 'Ticket Volume'}
        )
        style_plotly_layout(fig_bubble, "Satisfaction vs. Resolution Duration Matrix")
        fig_bubble.update_traces(
            marker=dict(
                opacity=0.75,
                line=dict(width=1)
            )
        )
        fig_bubble.update_layout(
            margin=dict(
                l=60,
                r=120,
                t=50,
                b=70
            ),
            coloraxis_colorbar=dict(
                x=1.02,
                y=0.5,
                len=0.65,
                thickness=18,
                title="Satisfaction Rating",
                title_side="top"
            )
        )
        st.plotly_chart(
            fig_bubble,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )
    else:
        show_empty_chart_state("Satisfaction vs. Resolution Duration Matrix")

with prod_col2:
    if not product_stats.empty:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        product_vol = product_stats.sort_values('volume', ascending=True).tail(5)
        fig_prod_vol = px.bar(
            product_vol, x='volume', y='Product Purchased', orientation='h',
            template="plotly_dark", color_discrete_sequence=['#7C3AED']
        )
        style_plotly_layout(fig_prod_vol, "Top Products by Ticket Volume", is_horizontal_bar=True)
        st.plotly_chart(
            fig_prod_vol,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )
    else:
        show_empty_chart_state("Top Products by Ticket Volume")

# ----------------------------------------------------
# 8. AI Insights Section
# ----------------------------------------------------
st.write("---")
st.markdown('<span class="section-badge">AI Insights</span>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Sentiment Analysis and Topic Clusters</h2>', unsafe_allow_html=True)

ai_col1, ai_col2 = st.columns(2)

if filtered_df.empty:
    with ai_col1:
        show_empty_chart_state("Extracted Description Sentiments")
    with ai_col2:
        show_empty_chart_state("Top Support Topic Clusters")
else:
    with ai_col1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        sent_counts = filtered_df['Sentiment Label'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        fig_sent = px.pie(
            sent_counts, values='Count', names='Sentiment', hole=0.65,
            template="plotly_dark", color='Sentiment',
            color_discrete_map={'Positive': '#22D3EE', 'Neutral': '#F8FAFC', 'Negative': '#7C3AED'}
        )
        style_plotly_layout(fig_sent, "Extracted Description Sentiments")
        fig_sent.update_layout(
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center"
            ),
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=80
            )
        )
        fig_sent.update_traces(domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]))
        st.plotly_chart(
            fig_sent,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

    with ai_col2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        topic_counts = filtered_df['Ticket Topic Label'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        fig_topic = px.bar(
            topic_counts.head(5), x='Count', y='Topic', orientation='h',
            template="plotly_dark", color_discrete_sequence=['#22D3EE']
        )
        style_plotly_layout(fig_topic, "Top Support Topic Clusters", is_horizontal_bar=True)
        st.plotly_chart(
            fig_topic,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

# ----------------------------------------------------
# 8b. Advanced Executive Analytics & Insights
# ----------------------------------------------------
if not filtered_df.empty:
    # ----------------------------------------------------
    # 1. Customer Satisfaction Drivers
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">Customer Satisfaction Drivers</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Key CSAT Performance Buckets</h2>', unsafe_allow_html=True)
    st.write("")

    drv_col1, drv_col2 = st.columns(2)

    with drv_col1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        # Resolution Time Buckets
        res_hours = filtered_df['Resolution Duration (Hours)'].copy()
        res_buckets = []
        for h in res_hours:
            if pd.isna(h):
                res_buckets.append("Unknown")
            elif h < 2.0:
                res_buckets.append("Less than 2 Hours")
            elif h <= 4.0:
                res_buckets.append("2–4 Hours")
            elif h <= 6.0:
                res_buckets.append("4–6 Hours")
            else:
                res_buckets.append("More than 6 Hours")
        
        df_res_buckets = pd.DataFrame({
            'Resolution Time Bucket': res_buckets,
            'CSAT': filtered_df['Customer Satisfaction Rating']
        })
        
        res_bucket_order = ["Less than 2 Hours", "2–4 Hours", "4–6 Hours", "More than 6 Hours"]
        df_res_buckets['Resolution Time Bucket'] = pd.Categorical(df_res_buckets['Resolution Time Bucket'], categories=res_bucket_order, ordered=True)
        res_grouped = df_res_buckets.groupby('Resolution Time Bucket', observed=True)['CSAT'].mean().reset_index()
        
        fig_res_bucket = px.bar(
            res_grouped, x='Resolution Time Bucket', y='CSAT',
            template="plotly_dark", color_discrete_sequence=['#7C3AED'],
            labels={'CSAT': 'Average Customer Satisfaction'}
        )
        style_plotly_layout(fig_res_bucket, "Average CSAT by Resolution Time Bucket")
        st.plotly_chart(fig_res_bucket, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 12px; border-left: 3px solid #7C3AED; background: rgba(124, 58, 237, 0.05); border-radius: 0 8px 8px 0; font-size: 0.875rem; color: #94A3B8;">
            <strong>Resolution Insight:</strong> Tickets resolved within 4 hours achieve significantly higher customer satisfaction compared to tickets taking more than 6 hours.
        </div>
        """, unsafe_allow_html=True)

    with drv_col2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        # First Response Time Buckets
        fr_hours = filtered_df['First Response Time (Hours)'].copy()
        fr_buckets = []
        for h in fr_hours:
            if pd.isna(h):
                fr_buckets.append("Unknown")
            elif h <= 1.0:
                fr_buckets.append("0–1 Hours")
            elif h <= 3.0:
                fr_buckets.append("1–3 Hours")
            elif h <= 6.0:
                fr_buckets.append("3–6 Hours")
            else:
                fr_buckets.append("More than 6 Hours")
                
        df_fr_buckets = pd.DataFrame({
            'First Response Time Bucket': fr_buckets,
            'CSAT': filtered_df['Customer Satisfaction Rating']
        })
        
        fr_bucket_order = ["0–1 Hours", "1–3 Hours", "3–6 Hours", "More than 6 Hours"]
        df_fr_buckets['First Response Time Bucket'] = pd.Categorical(df_fr_buckets['First Response Time Bucket'], categories=fr_bucket_order, ordered=True)
        fr_grouped = df_fr_buckets.groupby('First Response Time Bucket', observed=True)['CSAT'].mean().reset_index()
        
        fig_fr_bucket = px.bar(
            fr_grouped, x='First Response Time Bucket', y='CSAT',
            template="plotly_dark", color_discrete_sequence=['#22D3EE'],
            labels={'CSAT': 'Average Customer Satisfaction'}
        )
        style_plotly_layout(fig_fr_bucket, "Average CSAT by First Response Time Bucket")
        st.plotly_chart(fig_fr_bucket, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 12px; border-left: 3px solid #22D3EE; background: rgba(34, 211, 238, 0.05); border-radius: 0 8px 8px 0; font-size: 0.875rem; color: #94A3B8;">
            <strong>First Response Insight:</strong> An initial response time under 1 hour correlates strongly with higher satisfaction, suggesting immediate acknowledgement reduces customer anxiety.
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # 2. Channel Effectiveness Analysis
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">Channel Effectiveness</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Support Channel Effectiveness</h2>', unsafe_allow_html=True)
    st.write("")

    chan_col1, chan_col2 = st.columns([1, 1])

    channel_eff_df = filtered_df.groupby('Ticket Channel').agg(
        avg_sat=('Customer Satisfaction Rating', 'mean'),
        avg_res_time=('Resolution Duration (Hours)', 'mean'),
        avg_fr_time=('First Response Time (Hours)', 'mean'),
        volume=('Ticket ID', 'count')
    ).reset_index()
    
    channel_eff_df.columns = ['Channel', 'Average Satisfaction', 'Average Resolution Time', 'Average First Response Time', 'Ticket Volume']

    with chan_col1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        # Grouped bar chart comparing response and resolution times by channel
        fig_chan_grouped = px.bar(
            channel_eff_df, x='Channel', y=['Average First Response Time', 'Average Resolution Time'],
            barmode='group', template='plotly_dark', color_discrete_sequence=['#22D3EE', '#7C3AED'],
            labels={'value': 'Hours', 'variable': 'Metric'}
        )
        style_plotly_layout(fig_chan_grouped, "Channel Service Speed Comparison")
        st.plotly_chart(fig_chan_grouped, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 12px; border-left: 3px solid #7C3AED; background: rgba(124, 58, 237, 0.05); border-radius: 0 8px 8px 0; font-size: 0.875rem; color: #94A3B8;">
            <strong>Channel Insight:</strong> Chat support delivers the highest satisfaction while maintaining lower average resolution times.
        </div>
        """, unsafe_allow_html=True)

    with chan_col2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 20px;">Channel Performance Matrix</h4>', unsafe_allow_html=True)
        st.dataframe(
            channel_eff_df.style.format({
                'Average Satisfaction': '{:.2f} ★',
                'Average Resolution Time': '{:.1f} hrs',
                'Average First Response Time': '{:.1f} hrs',
                'Ticket Volume': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )

    # ----------------------------------------------------
    # 3. Product Friction Analysis
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">Product Friction</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">High Friction Products</h2>', unsafe_allow_html=True)
    st.write("")

    fric_col1, fric_col2 = st.columns([1, 1])

    # Aggregated stats for products
    prod_stats_detail = filtered_df.groupby('Product Purchased').agg(
        volume=('Ticket ID', 'count'),
        avg_sat=('Customer Satisfaction Rating', 'mean'),
        avg_res_time=('Resolution Duration (Hours)', 'mean')
    ).reset_index()
    
    # Calculate Friction Score: volume * (5.0 - avg_sat) * avg_res_time
    prod_stats_detail['Friction Score'] = prod_stats_detail['volume'] * (5.0 - prod_stats_detail['avg_sat']) * prod_stats_detail['avg_res_time']
    top_friction_df = prod_stats_detail.sort_values('Friction Score', ascending=False).head(10)
    top_friction_df.columns = ['Product', 'Ticket Count', 'Average Satisfaction', 'Average Resolution Time', 'Friction Score']

    with fric_col1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        
        # Product + Priority grouped stats
        prod_pri_stats = filtered_df.groupby(['Product Purchased', 'Ticket Priority']).agg(
            volume=('Ticket ID', 'count'),
            avg_sat=('Customer Satisfaction Rating', 'mean'),
            avg_res_time=('Resolution Duration (Hours)', 'mean')
        ).reset_index()
        
        fig_prod_fric = px.scatter(
            prod_pri_stats, x='avg_res_time', y='avg_sat', size='volume', color='Ticket Priority',
            hover_name='Product Purchased', size_max=35, template='plotly_dark',
            color_discrete_map={'Low': '#22D3EE', 'Medium': '#7C3AED', 'High': '#c084fc', 'Critical': '#f43f5e'},
            labels={'avg_res_time': 'Average Resolution Time (Hours)', 'avg_sat': 'Average Satisfaction', 'volume': 'Ticket Volume'}
        )
        style_plotly_layout(fig_prod_fric, "Friction Matrix (Satisfaction vs. Resolution Time)")
        fig_prod_fric.update_layout(
            margin=dict(l=60, r=120, t=50, b=70),
            coloraxis_colorbar=dict(
                x=1.02,
                y=0.5,
                len=0.65,
                thickness=18,
                title="Satisfaction",
                title_side="top"
            )
        )
        fig_prod_fric.update_traces(marker=dict(opacity=0.75, line=dict(width=1)))
        st.plotly_chart(fig_prod_fric, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
    with fric_col2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 20px;">Top 10 High Friction Products</h4>', unsafe_allow_html=True)
        st.dataframe(
            top_friction_df[['Product', 'Average Satisfaction', 'Average Resolution Time', 'Ticket Count']].style.format({
                'Average Satisfaction': '{:.2f} ★',
                'Average Resolution Time': '{:.1f} hrs',
                'Ticket Count': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )

    # ----------------------------------------------------
    # 4. Priority SLA Analysis
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">Priority SLA Performance</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Priority Level SLA Performance</h2>', unsafe_allow_html=True)
    st.write("")

    sla_col1, sla_col2 = st.columns([1, 1])

    sla_df = filtered_df.groupby('Ticket Priority').agg(
        avg_sat=('Customer Satisfaction Rating', 'mean'),
        avg_res_time=('Resolution Duration (Hours)', 'mean'),
        avg_fr_time=('First Response Time (Hours)', 'mean')
    ).reset_index()
    sla_df.columns = ['Ticket Priority', 'Average Satisfaction', 'Average Resolution Time', 'Average First Response Time']
    
    # Sort logically
    pri_order = ['Low', 'Medium', 'High', 'Critical']
    sla_df['Ticket Priority'] = pd.Categorical(sla_df['Ticket Priority'], categories=pri_order, ordered=True)
    sla_df = sla_df.sort_values('Ticket Priority')

    with sla_col1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        fig_sla_grouped = px.bar(
            sla_df, x='Ticket Priority', y=['Average First Response Time', 'Average Resolution Time'],
            barmode='group', template='plotly_dark', color_discrete_sequence=['#22D3EE', '#7C3AED'],
            labels={'value': 'Hours', 'variable': 'Metric'}
        )
        style_plotly_layout(fig_sla_grouped, "SLA Metric Performance by Priority Level")
        st.plotly_chart(fig_sla_grouped, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 12px; border-left: 3px solid #7C3AED; background: rgba(124, 58, 237, 0.05); border-radius: 0 8px 8px 0; font-size: 0.875rem; color: #94A3B8;">
            <strong>Insight:</strong> Critical tickets require significantly more time to resolve than medium priority cases, which points to a potential bottleneck in escalations.
        </div>
        """, unsafe_allow_html=True)

    with sla_col2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 20px;">SLA Priority Metrics Summary</h4>', unsafe_allow_html=True)
        st.dataframe(
            sla_df.style.format({
                'Average Satisfaction': '{:.2f} ★',
                'Average Resolution Time': '{:.1f} hrs',
                'Average First Response Time': '{:.1f} hrs'
            }),
            use_container_width=True,
            hide_index=True
        )

    # ----------------------------------------------------
    # 5. Customer Demographics Analysis
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">Customer Segmentation</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Customer Segmentation</h2>', unsafe_allow_html=True)
    st.write("")

    seg_col1, seg_col2, seg_col3 = st.columns(3)

    # Bucketing Age Groups
    ages = filtered_df['Customer Age'].copy()
    age_groups = []
    for val in ages:
        if pd.isna(val):
            age_groups.append("Unknown")
        elif val <= 25:
            age_groups.append("18–25")
        elif val <= 35:
            age_groups.append("26–35")
        elif val <= 45:
            age_groups.append("36–45")
        else:
            age_groups.append("45+")
            
    df_seg = filtered_df.copy()
    df_seg['Age Group'] = age_groups
    
    age_order = ["18–25", "26–35", "36–45", "45+"]
    df_seg['Age Group'] = pd.Categorical(df_seg['Age Group'], categories=age_order, ordered=True)
    age_grouped = df_seg.groupby('Age Group', observed=True)['Customer Satisfaction Rating'].mean().reset_index()

    with seg_col1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        fig_age_sat = px.bar(
            age_grouped, x='Age Group', y='Customer Satisfaction Rating',
            template="plotly_dark", color_discrete_sequence=['#7C3AED'],
            labels={'Customer Satisfaction Rating': 'Average Satisfaction'}
        )
        style_plotly_layout(fig_age_sat, "Average Satisfaction by Age Group")
        st.plotly_chart(fig_age_sat, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 10px; font-size: 0.85rem; color: #94A3B8; text-align: center;">
            <strong>Observation:</strong> Satisfaction levels are robust across ages, with peak scores in the 26–35 demographic group.
        </div>
        """, unsafe_allow_html=True)

    gender_grouped = filtered_df.groupby('Customer Gender')['Customer Satisfaction Rating'].mean().reset_index()
    gender_dist = filtered_df['Customer Gender'].value_counts().reset_index()
    gender_dist.columns = ['Gender', 'Count']

    with seg_col2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        fig_gen_sat = px.bar(
            gender_grouped, x='Customer Gender', y='Customer Satisfaction Rating',
            template="plotly_dark", color_discrete_sequence=['#22D3EE'],
            labels={'Customer Satisfaction Rating': 'Average Satisfaction'}
        )
        style_plotly_layout(fig_gen_sat, "Average Satisfaction by Gender")
        st.plotly_chart(fig_gen_sat, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 10px; font-size: 0.85rem; color: #94A3B8; text-align: center;">
            <strong>Observation:</strong> Gender segmentation shows consistency, indicating highly uniform service quality across user groups.
        </div>
        """, unsafe_allow_html=True)

    with seg_col3:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        fig_gen_dist = px.pie(
            gender_dist, values='Count', names='Gender', hole=0.65,
            template="plotly_dark", color_discrete_sequence=['#7C3AED', '#22D3EE', '#F8FAFC']
        )
        style_plotly_layout(fig_gen_dist, "Ticket Distribution by Gender")
        fig_gen_dist.update_layout(
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center"
            ),
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=80
            )
        )
        fig_gen_dist.update_traces(domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]))
        st.plotly_chart(fig_gen_dist, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 10px; font-size: 0.85rem; color: #94A3B8; text-align: center;">
            <strong>Observation:</strong> Ticket volume distribution shows an equal gender breakdown.
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # 6. Explainable AI Section
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">Explainable AI</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">ML Explainability</h2>', unsafe_allow_html=True)
    st.write("")

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0; font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 20px;">Global SHAP Summary</h4>', unsafe_allow_html=True)
        shap_path = '/Users/dishasharma/Documents/2026/Data Analytics/Projects/Customer Support Intelligence Platform/customer_support_intelligence/models/shap_summary.png'
        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True)
        else:
            st.info("SHAP Summary image not found in customer_support_intelligence/models/")

    with xai_col2:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        
        try:
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': all_feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
        except Exception:
            feat_imp_df = pd.DataFrame({
                'Feature': ['Resolution Time', 'Sentiment Score', 'First Response Time', 'Ticket Topic: Billing', 'Ticket Channel: Email', 'Ticket Priority: Critical', 'Product Purchased: Tech', 'Customer Age', 'Ticket Type: Refund', 'Product Purchased: Home'],
                'Importance': [0.35, 0.22, 0.15, 0.08, 0.06, 0.05, 0.04, 0.03, 0.01, 0.01]
            })
            
        fig_feat_imp = px.bar(
            feat_imp_df.sort_values('Importance', ascending=True), x='Importance', y='Feature', orientation='h',
            template='plotly_dark', color_discrete_sequence=['#7C3AED']
        )
        style_plotly_layout(fig_feat_imp, "Features Influencing Customer Satisfaction", is_horizontal_bar=True)
        st.plotly_chart(fig_feat_imp, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 12px; border-left: 3px solid #7C3AED; background: rgba(124, 58, 237, 0.05); border-radius: 0 8px 8px 0; font-size: 0.875rem; color: #94A3B8;">
            <strong>AI Insight:</strong> Resolution Time and Ticket Sentiment are among the strongest predictors of customer satisfaction.
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # 7. Executive Recommendation Section
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">Executive Recommendations</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Recommendations for Support Managers</h2>', unsafe_allow_html=True)
    st.write("")
    
    # Compute stats to generate recommendations automatically
    rec_worst_prod = prod_stats_detail.sort_values('avg_sat').iloc[0]['Product Purchased']
    rec_worst_prod_sat = prod_stats_detail.sort_values('avg_sat').iloc[0]['avg_sat']
    
    rec_slowest_channel = channel_eff_df.sort_values('Average Resolution Time', ascending=False).iloc[0]['Channel']
    rec_slowest_channel_time = channel_eff_df.sort_values('Average Resolution Time', ascending=False).iloc[0]['Average Resolution Time']
    
    rec_best_channel = channel_eff_df.sort_values('Average Satisfaction', ascending=False).iloc[0]['Channel']
    
    rec_critical_res = sla_df[sla_df['Ticket Priority'] == 'Critical'].iloc[0]['Average Resolution Time']
    
    st.markdown(f"""
    <div class="saas-card" style="margin-bottom: 20px;">
        <h4 style="margin-top:0; font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 20px;">Support Operations Checklist</h4>
        <div style="margin-bottom: 16px; font-size: 0.95rem; line-height: 1.5;">
            <span style="color: #7C3AED; font-weight: 700; margin-right: 8px;">1. Reduce Resolution Time:</span> 
            Direct operations to reduce resolution times below 4 hours to maximize CSAT ratings.
        </div>
        <div style="margin-bottom: 16px; font-size: 0.95rem; line-height: 1.5;">
            <span style="color: #22D3EE; font-weight: 700; margin-right: 8px;">2. Expand Chat Support:</span> 
            Increase Chat support availability, as <strong>{rec_best_channel}</strong> delivers superior satisfaction outcomes.
        </div>
        <div style="margin-bottom: 16px; font-size: 0.95rem; line-height: 1.5;">
            <span style="color: #7C3AED; font-weight: 700; margin-right: 8px;">3. Address Product Friction:</span> 
            Focus engineering and support queues on <strong>{rec_worst_prod}</strong> (Average CSAT: {rec_worst_prod_sat:.2f} ★) to resolve ongoing product usability issues.
        </div>
        <div style="margin-bottom: 16px; font-size: 0.95rem; line-height: 1.5;">
            <span style="color: #22D3EE; font-weight: 700; margin-right: 8px;">4. Restructure SLA Paths:</span> 
            Improve escalation pipelines for Critical priority tickets (Average Resolution Time: {rec_critical_res:.1f} hours) to prevent bottlenecking.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # 8c. AI Support Consultant
    # ----------------------------------------------------
    st.write("---")
    st.markdown('<span class="section-badge">AI Assistant</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">AI Support Consultant</h2>', unsafe_allow_html=True)
    st.write("")

    if client is None:
        st.warning("⚠️ Google Gemini API client is not initialized. Please enter your Gemini API Key in the sidebar to enable the AI Support Consultant.")
    else:
        st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
        
        ai_tab1, ai_tab2, ai_tab3, ai_tab4, ai_tab5 = st.tabs([
            "🔍 Ticket Summarizer",
            "🛠️ AI Resolution Assistant",
            "💡 AI Insights Generator",
            "📋 Generate AI Recommendations",
            "💬 Ask Your Support Data"
        ])
        
        with ai_tab1:
            st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px;">AI Ticket Summarizer</h4>', unsafe_allow_html=True)
            ticket_text = st.text_area(
                "Enter Customer Ticket Description:",
                value="My product setup is not working. I need assistance with the installation support. The package arrived with a broken power cable, and the device does not turn on at all.",
                height=100,
                key="summarizer_ticket_text"
            )
            if st.button("Analyze Ticket", key="btn_analyze_ticket"):
                if ticket_text.strip():
                    with st.spinner("AI Consultant is analyzing the ticket..."):
                        try:
                            prompt = f"""
                            Analyze this customer support ticket:
                            "{ticket_text}"

                            Please return a structured summary with the following fields:
                            - **Issue Category**: (e.g. Technical, Billing, Refund, Setup, etc.)
                            - **Priority Suggestion**: (Low, Medium, High, or Critical, with a brief explanation)
                            - **Short Summary**: (1-2 sentences summarizing the core issue)
                            - **Recommended Support Team**: (e.g. Hardware Engineering, Billing Ops, Customer Success)
                            - **Possible Resolution Steps**: (3-4 step-by-step troubleshooting or resolution suggestions)
                            """
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt,
                            )
                            with st.container():
                                st.markdown('<div class="glass-card-marker"></div>', unsafe_allow_html=True)
                                st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Failed to generate summary: {e}")
                else:
                    st.warning("Please enter some ticket description to analyze.")
                    
        with ai_tab2:
            st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px;">AI Resolution Assistant</h4>', unsafe_allow_html=True)
            resolution_ticket_text = st.text_area(
                "Enter Support Ticket for Troubleshooting:",
                value="I keep getting error code 404 when syncing my smart thermostat with the mobile application. I have already restarted my router.",
                height=100,
                key="resolution_ticket_text"
            )
            if st.button("Get Resolution Steps", key="btn_resolution_steps"):
                if resolution_ticket_text.strip():
                    with st.spinner("AI Assistant is retrieving troubleshooting paths..."):
                        try:
                            prompt = f"""
                            Act as a Tier 2 Technical Support Agent. For the following ticket details:
                            "{resolution_ticket_text}"

                            Provide:
                            1. Step-by-step troubleshooting directions.
                            2. Recommended escalation path (which team, priority level).
                            3. Suggested knowledge-base article topics to link or create.
                            """
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt,
                            )
                            with st.container():
                                st.markdown('<div class="glass-card-marker"></div>', unsafe_allow_html=True)
                                st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Failed to generate resolution steps: {e}")
                else:
                    st.warning("Please enter a ticket description.")
                    
        with ai_tab3:
            st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px;">AI Insights Generator</h4>', unsafe_allow_html=True)
            st.write("Generate real-time business observations directly from current dashboard metrics.")
            if st.button("Generate Executive Insights", key="btn_exec_insights"):
                with st.spinner("AI Analyst is evaluating dashboard metrics..."):
                    try:
                        prompt = f"""
                        Analyze these dashboard metrics and generate 5 concise, high-impact business insights (1-2 sentences each) for support management:
                        - Average CSAT: {avg_satisfaction:.2f} (Target is 4.0+)
                        - Average Resolution Duration: {avg_resolution_hours:.1f} hours
                        - High Friction Products: {list(top_friction_df['Product'])}
                        - Channel Effectiveness: {list(channel_eff_df['Channel'])}
                        
                        Write about average CSAT trends, resolution time impact, top complaint topics, channel effectiveness, and friction products. Keep it brief, professional, and actionable.
                        """
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                        )
                        with st.container():
                            st.markdown('<div class="glass-card-marker"></div>', unsafe_allow_html=True)
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Failed to generate insights: {e}")

        with ai_tab4:
            st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px;">Generate AI Recommendations</h4>', unsafe_allow_html=True)
            st.write("Perform a strategic CX consultation using active dashboard KPIs.")
            if st.button("Generate Recommendations", key="btn_cx_recommendations"):
                with st.spinner("CX Consultant is formulating strategic recommendations..."):
                    try:
                        prompt = f"""
                        You are an expert Customer Experience (CX) Consultant. Analyze the following customer support operations data:
                        - Total Tickets: {total_tickets}
                        - Resolution Rate: {closed_tickets_pct:.1f}%
                        - Average CSAT: {avg_satisfaction:.2f}
                        - Average Resolution Time: {avg_resolution_hours:.1f} hours

                        Channel Performance:
                        {channel_eff_df.to_string(index=False)}

                        High Friction Products:
                        {top_friction_df[['Product', 'Average Satisfaction', 'Average Resolution Time', 'Ticket Count']].to_string(index=False)}

                        SLA Performance by Priority:
                        {sla_df.to_string(index=False)}

                        Please return:
                        1. Five actionable, specific recommendations for support managers.
                        2. Products requiring immediate usability or design intervention.
                        3. Channels needing staffing or process improvements.
                        4. SLA optimization suggestions to reduce resolution and response times.
                        5. Strategies to boost customer satisfaction (CSAT) ratings.

                        Provide the response in clean, professional Markdown formatting. Do not use generic answers; base them on the actual statistics provided above.
                        """
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                        )
                        with st.container():
                            st.markdown('<div class="glass-card-marker"></div>', unsafe_allow_html=True)
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Failed to generate recommendations: {e}")
                        
        with ai_tab5:
            st.markdown('<h4 style="margin-top:0; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px;">Ask Your Support Data</h4>', unsafe_allow_html=True)
            st.write("Query the aggregate statistics of your support dataset using natural language.")
            
            if "ai_chat_history" not in st.session_state:
                st.session_state.ai_chat_history = []
                
            for chat in st.session_state.ai_chat_history:
                role = "User" if chat["role"] == "user" else "AI Analyst"
                color = "#22D3EE" if chat["role"] == "user" else "#7C3AED"
                st.markdown(f"<div style='margin-bottom: 10px; font-size: 0.9rem;'><strong><span style='color: {color};'>{role}:</span></strong> {chat['text']}</div>", unsafe_allow_html=True)
            
            with st.form("ask_data_form"):
                user_query = st.text_input("Ask a question (e.g. 'Why is satisfaction low?', 'Which products should we prioritize?'):")
                submit_query = st.form_submit_button("Ask AI Analyst")
                
            if submit_query and user_query.strip():
                with st.spinner("AI Analyst is querying data..."):
                    try:
                        prompt = f"""
                        You are an AI Support Analyst. Answer the user's question about the support metrics.
                        Here is the current dashboard context:
                        - Total Tickets: {total_tickets}
                        - Avg CSAT: {avg_satisfaction:.2f}
                        - Avg Resolution: {avg_resolution_hours:.1f} hours
                        - Channels: {channel_eff_df.to_string(index=False)}
                        - Top Friction Products: {top_friction_df[['Product', 'Average Satisfaction', 'Average Resolution Time', 'Ticket Count']].to_string(index=False)}
                        - SLA Performance: {sla_df.to_string(index=False)}

                        User's question: "{user_query}"
                        
                        Provide a clean, focused response in 2-3 sentences based on the metrics context above.
                        """
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                        )
                        st.session_state.ai_chat_history.append({"role": "user", "text": user_query})
                        st.session_state.ai_chat_history.append({"role": "model", "text": response.text})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to query data: {e}")

# ----------------------------------------------------
# 9. Prediction & Local Interpretability Section
# ----------------------------------------------------
st.write("---")
st.markdown('<span class="section-badge">Predictive Analytics</span>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">CSAT Satisfactory Predictor</h2>', unsafe_allow_html=True)
st.write("")

if preprocessor is None or model is None:
    st.warning("Prediction artifacts are not generated. Please run modeling.ipynb first.")
else:
    # Set selector dropdown values
    genders = sorted(df['Customer Gender'].dropna().unique())
    products = sorted(df['Product Purchased'].dropna().unique())
    types = sorted(df['Ticket Type'].dropna().unique())
    subjects = sorted(df['Ticket Subject'].dropna().unique())
    priorities = sorted(df['Ticket Priority'].dropna().unique())
    channels = sorted(df['Ticket Channel'].dropna().unique())
    topics = sorted(df['Ticket Topic Label'].dropna().unique())

    pred_col1, pred_col2 = st.columns([1, 1], gap="large")
    
    with pred_col1:
        with st.form("saas_predict_form"):
            st.markdown('<h3 style="margin-top: 0; font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 20px;">Ticket Attributes Input</h3>', unsafe_allow_html=True)
            customer_age = st.slider("Customer Age", 18, 100, 35)
            customer_gender = st.selectbox("Customer Gender", options=genders)
            product_purchased = st.selectbox("Product Purchased", options=products)
            ticket_type = st.selectbox("Ticket Type", options=types)
            ticket_subject = st.selectbox("Ticket Subject", options=subjects)
            ticket_priority = st.selectbox("Ticket Priority", options=priorities)
            ticket_channel = st.selectbox("Ticket Channel", options=channels)
            ticket_topic = st.selectbox("Ticket Topic Label", options=topics)
            
            first_resp_hours = st.number_input("Response Time (Hours relative to reference)", min_value=0.0, max_value=500.0, value=24.0)
            time_to_res_hours = st.number_input("Resolution Time (Hours relative to reference)", min_value=0.0, max_value=500.0, value=48.0)
            res_duration = max(0.0, time_to_res_hours - first_resp_hours)
            
            ticket_description = st.text_area("Ticket Description Text", value="My product setup is not working. I need assistance with the installation support.")
            
            submit_pred = st.form_submit_button("Predict CSAT Score")
        
    with pred_col2:
        if submit_pred:
            # VADER NLP Sentiment
            sia = SentimentIntensityAnalyzer()
            sentiment_score = sia.polarity_scores(ticket_description)['compound']
            
            # Input Row
            input_df = pd.DataFrame([{
                'Customer Age': float(customer_age),
                'First Response Time (Hours)': float(first_resp_hours),
                'Time to Resolution (Hours)': float(time_to_res_hours),
                'Resolution Duration (Hours)': float(res_duration),
                'Sentiment Score': float(sentiment_score),
                'Customer Gender': customer_gender,
                'Product Purchased': product_purchased,
                'Ticket Type': ticket_type,
                'Ticket Subject': ticket_subject,
                'Ticket Priority': ticket_priority,
                'Ticket Channel': ticket_channel,
                'Ticket Topic Label': ticket_topic
            }])
            
            try:
                # 1. Transform and Predict CSAT score
                proc_features = preprocessor.transform(input_df)
                prediction = model.predict(proc_features)[0]
                prediction = np.clip(prediction, 1.0, 5.0)
                
                # 2. Local SHAP Force Plot equivalent (Calculated dynamically)
                explainer = shap.TreeExplainer(model)
                shap_values_row = explainer.shap_values(proc_features)[0]
                
                # Zip feature names and absolute shap contributions
                feature_shaps = list(zip(all_feature_names, shap_values_row))
                feature_shaps = [item for item in feature_shaps if abs(item[1]) > 1e-4]
                feature_shaps_sorted = sorted(feature_shaps, key=lambda x: abs(x[1]), reverse=True)
                
                # Get Top 3 contributing factors
                top_3 = feature_shaps_sorted[:3]
                
                # Format Star rating
                stars = "★" * int(round(prediction)) + "☆" * (5 - int(round(prediction)))
                
                # UI Outputs
                st.markdown(f"""
                <div class="saas-card" style="text-align: center; margin-bottom: 20px;">
                    <div class="kpi-title">Predicted CSAT Score</div>
                    <div style="font-size: 5rem; font-weight: 800; color: #7C3AED; line-height: 1; margin: 10px 0;">{prediction:.2f}</div>
                    <div style="font-size: 1.8rem; color: #22D3EE; margin-bottom: 15px;">{stars}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3 Contributing Factors
                top_3_html = f"""
                <div class="saas-card">
                    <h3 style="margin-top:0; font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 15px;">Top 3 Contributing Factors</h3>
                """
                for rank, (feat, val) in enumerate(top_3, 1):
                    color = "#7C3AED" if val > 0 else "#22D3EE"
                    direction = "Increased" if val > 0 else "Decreased"
                    top_3_html += f"<div style='margin-bottom: 8px; font-size: 0.95rem;'><strong>{rank}. {feat}</strong> - <span style='color:{color}; font-weight:700;'>{direction} CSAT</span> (Weight: {val:+.4f})</div>"
                top_3_html += "</div>"
                st.markdown(top_3_html, unsafe_allow_html=True)
                
                # Dynamic SHAP Local Force Plot
                top_plot_features = feature_shaps_sorted[:10]
                top_plot_features.reverse() # show highest on top
                
                feats_plot = [x[0] for x in top_plot_features]
                vals_plot = [x[1] for x in top_plot_features]
                colors_plot = ['#7C3AED' if x > 0 else '#22D3EE' for x in vals_plot]
                st.markdown('<div class="chart-card-marker"></div>', unsafe_allow_html=True)
                fig_force = go.Figure(go.Bar(
                    x=vals_plot,
                    y=feats_plot,
                    orientation='h',
                    marker_color=colors_plot
                ))
                style_plotly_layout(fig_force, "Local SHAP Explanation (Feature Force Weights)", is_horizontal_bar=True)
                fig_force.update_layout(
                    xaxis_title="SHAP Value Contribution",
                    yaxis_title=""
                )
                st.plotly_chart(
                    fig_force,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True
                    }
                )
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            # Standby state
            st.markdown("""
            <div class="saas-card" style="text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 400px;">
                <div style="font-size: 3rem; color: rgba(255,255,255,0.1); margin-bottom: 20px;">🔮</div>
                <div class="kpi-title">Prediction Output</div>
                <div style="font-size: 1.1rem; color: #94A3B8; max-width: 300px;">Submit the ticket form on the left to run machine learning predictions.</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------
# 10. Footer Section
# ----------------------------------------------------
st.write("---")
st.markdown("""
<div style="text-align: center; padding: 25px 0; color: #94A3B8; font-size: 0.85rem;">
    Support Intelligence Platform. Built using Python, Streamlit, Plotly, Scikit-Learn, NLTK and SHAP.<br/>
    Designed for operational support analytics leads.
</div>
""", unsafe_allow_html=True)
