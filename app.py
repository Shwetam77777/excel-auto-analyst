import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM, Groq
import matplotlib.pyplot as plt

# 1. Configuration
st.set_page_config(
    page_title="Excel Auto-Analyst", 
    page_icon="📊", 
    layout="wide"
)

# 2. Sidebar Setup (The Navigation)
with st.sidebar:
    st.title("📊 Auto-Analyst")
    st.write("Upload your data and navigate through the tabs below.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=['csv', 'xlsx'])
    
    # Navigation Menu
    page = st.radio("Navigate to:", ["🏠 Home & Data Cleaning", "📈 Auto-Dashboard", "🎨 Custom Analysis", "🗣️ Chat with Data"])
    
    st.info("Built with Streamlit & Python")

# Function to load data (Cached to prevent reloading on every click)
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# 3. Main Application Logic
if uploaded_file is not None:
    # Load Data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Detect Columns
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        # --- PAGE 1: HOME & DATA CLEANING ---
        if page == "🏠 Home & Data Cleaning":
            st.title("🏠 Data Overview & Cleaning")
            st.markdown("### 1. Raw Data Preview")
            st.dataframe(df.head())

            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())

            st.markdown("---")
            st.markdown("### 2. Auto-Cleaning Options")
            
            clean_mode = st.checkbox("✅ Enable Auto-Cleaning Mode")
            
            if clean_mode:
                df_cleaned = df.copy()
                # Remove duplicates
                df_cleaned = df_cleaned.drop_duplicates()
                # Fill missing numbers with 0 (safe default)
                df_cleaned[num_cols] = df_cleaned[num_cols].fillna(0)
                # Fill missing text with "Unknown"
                df_cleaned[cat_cols] = df_cleaned[cat_cols].fillna("Unknown")
                
                st.success("Data Cleaned! Duplicates removed and missing values filled.")
                st.dataframe(df_cleaned.head())
                
                # Download Button
                csv = df_cleaned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Data",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                )
                
                # Save cleaned data to session state so other pages can use it
                st.session_state['df_cleaned'] = df_cleaned
            else:
                # If not cleaning, use original data
                st.session_state['df_cleaned'] = df

        # --- PAGE 2: AUTO-DASHBOARD ---
        elif page == "📈 Auto-Dashboard":
            st.title("📈 Instant Insights Dashboard")
            
            # Retrieve data from session state
            if 'df_cleaned' in st.session_state:
                df_active = st.session_state['df_cleaned']
                
                if len(num_cols) > 0:
                    # KPI Cards
                    st.subheader("Key Performance Indicators")
                    # Pick the first numeric column as the "Primary Metric" (e.g., Sales)
                    metric_col = st.selectbox("Select Key Metric for KPIs:", num_cols, index=0)
                    
                    total = df_active[metric_col].sum()
                    avg = df_active[metric_col].mean()
                    maxx = df_active[metric_col].max()
                    
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Total Sum", f"{total:,.2f}")
                    kpi2.metric("Average", f"{avg:,.2f}")
                    kpi3.metric("Max Value", f"{maxx:,.2f}")
                    
                    st.markdown("---")
                    
                    # Auto-Charts
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        st.subheader("Distribution")
                        fig_hist = px.histogram(df_active, x=metric_col, title=f"Distribution of {metric_col}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                    with col_chart2:
                        if len(cat_cols) > 0:
                            st.subheader("Categorical Split")
                            cat_col = st.selectbox("Select Category:", cat_cols, key="dash_cat")
                            df_grouped = df_active.groupby(cat_col)[metric_col].sum().reset_index()
                            fig_pie = px.pie(df_grouped, names=cat_col, values=metric_col, title=f"{metric_col} by {cat_col}")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.info("No text columns found for categorical analysis.")
                else:
                    st.warning("No numeric columns found to generate dashboards.")
            else:
                st.warning("Please go to the Home tab and check 'Enable Auto-Cleaning' first.")

        # --- PAGE 3: CUSTOM ANALYSIS & AI ---
        elif page == "🎨 Custom Analysis":
            st.title("🎨 Custom Report Builder")
            
            if 'df_cleaned' in st.session_state:
                df_active = st.session_state['df_cleaned']
                
                # Controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_axis = st.selectbox("X-Axis (Category/Time)", df_active.columns)
                with col2:
                    y_axis = st.selectbox("Y-Axis (Values)", num_cols)
                with col3:
                    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
                
                # Generate Button
                if st.button("Generate Analysis"):
                    st.markdown("---")
                    
                    # Chart Logic
                    if chart_type == "Bar Chart":
                        df_grouped = df_active.groupby(x_axis)[y_axis].sum().reset_index()
                        fig = px.bar(df_grouped, x=x_axis, y=y_axis, color=y_axis)
                    elif chart_type == "Line Chart":
                        df_sorted = df_active.sort_values(by=x_axis)
                        fig = px.line(df_sorted, x=x_axis, y=y_axis)
                    else:
                        fig = px.scatter(df_active, x=x_axis, y=y_axis, color=x_axis)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI Insights Logic
                    st.subheader("🤖 AI Insights")
                    
                    # Calculate simple stats for narrative
                    max_val = df_active[y_axis].max()
                    min_val = df_active[y_axis].min()
                    
                    # Basic Trend Analysis
                    try:
                        start_val = df_active.sort_values(by=x_axis)[y_axis].iloc[0]
                        end_val = df_active.sort_values(by=x_axis)[y_axis].iloc[-1]
                        if end_val > start_val:
                            trend = "increasing 📈"
                        elif end_val < start_val:
                            trend = "decreasing 📉"
                        else:
                            trend = "stable ➖"
                    except:
                        trend = "fluctuating"

                    insight = f"""
                    * **Observation:** The values for **{y_axis}** range from **{min_val:,.2f}** to **{max_val:,.2f}**.
                    * **Trend:** Over the course of **{x_axis}**, the data appears to be **{trend}**.
                    * **Peak:** The highest point helps identify the most performing category or time period.
                    """
                    st.info(insight)
                    
            else:
                st.warning("Please go to the Home tab and check 'Enable Auto-Cleaning' first.")

        # --- PAGE 4: CHAT WITH DATA (GEN-AI) ---
        elif page == "🗣️ Chat with Data":
            st.title("🗣️ Chat with your Data")
            st.markdown("ask questions in plain English and let AI generate insights & charts.")

            if 'df_cleaned' in st.session_state:
                df_active = st.session_state['df_cleaned']

                # API Key Handling: Check Secrets first, then Fallback to Input
                try:
                    # Try to retrieve from Streamlit Secrets (for Cloud Deployment)
                    groq_key = st.secrets["GROQ_API_KEY"]
                except (FileNotFoundError, KeyError):
                    # If not found, ask user for input
                    with st.expander("🔑 Setup: Enter LLM API Key", expanded=True):
                        st.info("Get your Free API Key from [Groq Cloud](https://console.groq.com/keys).")
                        groq_key = st.text_input("Groq API Key", type="password", help="The system uses Llama3-70b for fast analysis.")

                if groq_key:
                    # Initialize LLM
                    try:
                        llm = Groq(api_key=groq_key, model="llama3-70b-8192")
                        sdf = SmartDataframe(df_active, config={"llm": llm})
                        
                        # Chat Interface
                        prompt = st.chat_input("Ask something (e.g., 'Plot top 5 sales by region')")
                        
                        if prompt:
                            with st.spinner("🤖 Thinking..."):
                                result = sdf.chat(prompt)
                                
                                st.write("**Answer:**")
                                st.write(result)
                                
                                # PandasAI usually returns a path to a chart if generated
                                # For Streamlit, it renders automatically if it's an image path, but we can double check
                                if isinstance(result, str) and result.endswith(".png"):
                                    st.image(result)
                                    
                    except Exception as e:
                        st.error(f"Error initializing AI: {e}")
                else:
                    st.warning("⚠️ Please enter a valid API Key to start chatting.")
            
            else:
                st.warning("Please go to the Home tab and check 'Enable Auto-Cleaning' first.")

else:
    # Landing Page when no file is uploaded
    st.info("👈 Please upload a CSV or Excel file from the sidebar to begin.")
    st.markdown("""
    ### Welcome to Excel Auto-Analyst!
    This app helps you:
    1. **Clean Data** automatically (remove duplicates, fill missing values).
    2. **Visualize** trends with instant dashboards.
    3. **Analyze** custom relationships with AI-powered summaries.
    """) 
