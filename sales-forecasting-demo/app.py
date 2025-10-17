import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import openai
import os
from dotenv import load_dotenv
import json
import re
import hashlib

# Load environment variables
load_dotenv()

# Simple authentication system
USERS = {
    'admin': {
        'password': 'Admin123!',
        'role': 'admin',
        'name': 'Admin User'
    },
    'customer': {
        'password': 'Customer123!',
        'role': 'customer', 
        'name': 'Customer User'
    }
}

def check_password(username, password):
    """Check if username and password are correct"""
    if username in USERS and USERS[username]['password'] == password:
        return True
    return False

def get_user_role(username):
    """Get user role"""
    return USERS.get(username, {}).get('role', 'customer')

def get_user_name(username):
    """Get user display name"""
    return USERS.get(username, {}).get('name', username)

# Page configuration
st.set_page_config(
    page_title="AI Sales Forecasting Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    if 'selected_date_col' not in st.session_state:
        st.session_state.selected_date_col = None
    if 'selected_target_col' not in st.session_state:
        st.session_state.selected_target_col = None
    if 'selected_product_col' not in st.session_state:
        st.session_state.selected_product_col = None
    if 'selected_customer_col' not in st.session_state:
        st.session_state.selected_customer_col = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ai_detected_mapping' not in st.session_state:
        st.session_state.ai_detected_mapping = None
    # Authentication-related session state
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'user_data_path' not in st.session_state:
        st.session_state.user_data_path = None

def setup_openai():
    """Setup OpenAI client with error handling"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_key_here':
            client = openai.OpenAI(api_key=api_key)
            return True, "OpenAI API configured successfully"
        else:
            return False, "OpenAI API key not configured"
    except Exception as e:
        return False, f"Error setting up OpenAI: {str(e)}"

def generate_ai_response(question, forecast_data):
    """Generate AI response using OpenAI API"""
    try:
        # Check if OpenAI is configured
        is_configured, message = setup_openai()
        if not is_configured:
            return f"⚠️ {message}. Please configure your OpenAI API key in the .env file."
        
        # Prepare context
        trend_direction = "increasing" if forecast_data['slope'] > 0 else "decreasing"
        trend_strength = abs(forecast_data['slope'])
        
        context = f"""
        You are an AI sales forecasting expert for food manufacturing. 
        
        Forecast Data:
        - Target Column: {forecast_data['target_col']}
        - Trend: {trend_direction} (strength: {trend_strength:.4f})
        - Forecast Period: {forecast_data['periods']} {forecast_data.get('period_label', 'days')}
        - Historical Data Points: {len(forecast_data['historical_data'])}
        - Average Forecast Value: {sum(forecast_data['forecast_values'])/len(forecast_data['forecast_values']):.2f}
        - Forecast Range: {min(forecast_data['forecast_values']):.2f} to {max(forecast_data['forecast_values']):.2f}
        
        Question: {question}
        
        Provide business-focused insights about this sales forecast. Be specific, actionable, and explain patterns in terms of seasonality, market trends, and operational planning. Always mention which forecasting model was used and why it's appropriate.
        """
        
        # Call OpenAI API using new client format
        api_key = os.getenv('OPENAI_API_KEY')
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI sales forecasting expert for food manufacturing. Provide business-focused insights about sales forecasts. Be specific, actionable, and explain patterns in terms of seasonality, market trends, and operational planning."},
                {"role": "user", "content": context}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error generating AI response: {str(e)}"

def aggregate_data(df, date_col, target_col, aggregation):
    """Aggregate data by specified time period"""
    df_agg = df.copy()
    df_agg[date_col] = pd.to_datetime(df_agg[date_col])
    
    if aggregation == "Daily":
        return df_agg.groupby(date_col)[target_col].sum().reset_index()
    elif aggregation == "Weekly":
        df_agg['week'] = df_agg[date_col].dt.to_period('W')
        result = df_agg.groupby('week')[target_col].sum().reset_index()
        result['week'] = result['week'].dt.to_timestamp()
        return result.rename(columns={'week': date_col})
    elif aggregation == "Monthly":
        df_agg['month'] = df_agg[date_col].dt.to_period('M')
        result = df_agg.groupby('month')[target_col].sum().reset_index()
        result['month'] = result['month'].dt.to_timestamp()
        return result.rename(columns={'month': date_col})
    elif aggregation == "Yearly":
        df_agg['year'] = df_agg[date_col].dt.to_period('Y')
        result = df_agg.groupby('year')[target_col].sum().reset_index()
        result['year'] = result['year'].dt.to_timestamp()
        return result.rename(columns={'year': date_col})
    
    return df_agg

def get_confidence_multiplier(confidence_level):
    """Get z-score multiplier for confidence level"""
    confidence_multipliers = {
        80: 1.28,   # 80% confidence
        90: 1.645,  # 90% confidence  
        95: 1.96,   # 95% confidence
        99: 2.576   # 99% confidence
    }
    return confidence_multipliers.get(confidence_level, 1.96)

def linear_regression_forecast(data, date_col, target_col, periods, aggregation='Daily', annual_growth_rate=0.0, confidence_level=95):
    """Linear Regression forecasting model"""
    try:
        # Prepare data
        data_sorted = data.sort_values(date_col)
        X = np.arange(len(data_sorted)).reshape(-1, 1)
        y = data_sorted[target_col].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast
        future_X = np.arange(len(data_sorted), len(data_sorted) + periods).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        
        # Calculate confidence intervals
        residuals = y - model.predict(X)
        mse = np.mean(residuals**2)
        
        # Calculate prediction intervals
        confidence_multiplier = get_confidence_multiplier(confidence_level)
        
        # Standard error for prediction intervals
        x_mean = np.mean(X)
        sxx = np.sum((X - x_mean)**2)
        
        prediction_std = np.sqrt(mse * (1 + 1/len(X) + (future_X.flatten() - x_mean)**2 / sxx))
        
        # Calculate confidence bounds
        lower_bound = forecast_values - confidence_multiplier * prediction_std
        upper_bound = forecast_values + confidence_multiplier * prediction_std
        
        # Apply annual growth rate if specified
        if annual_growth_rate != 0.0:
            # Convert annual growth rate to period growth rate
            period_growth_rate = annual_growth_rate / 100.0
            
            # Calculate growth multiplier for each period
            if aggregation == "Daily":
                period_multiplier = (1 + period_growth_rate) ** (1/365)  # Daily growth
            elif aggregation == "Weekly":
                period_multiplier = (1 + period_growth_rate) ** (1/52)   # Weekly growth
            elif aggregation == "Monthly":
                period_multiplier = (1 + period_growth_rate) ** (1/12)   # Monthly growth
            elif aggregation == "Yearly":
                period_multiplier = (1 + period_growth_rate)             # Yearly growth
            else:
                period_multiplier = 1.0
            
            # Apply growth to forecast values and bounds
            for i in range(len(forecast_values)):
                growth_factor = period_multiplier ** (i + 1)
                forecast_values[i] *= growth_factor
                lower_bound[i] *= growth_factor
                upper_bound[i] *= growth_factor
        
        # Create forecast dates
        last_date = data_sorted[date_col].iloc[-1]
        if hasattr(last_date, 'to_timestamp'):
            last_date = last_date.to_timestamp()
        elif isinstance(last_date, pd.Period):
            last_date = last_date.to_timestamp()
        
        # Determine frequency based on aggregation
        freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME', 'Yearly': 'YE'}
        freq = freq_map.get(aggregation, 'D')
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq=freq)
        
        return {
            'forecast_values': forecast_values,
            'forecast_dates': forecast_dates,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'model_name': 'Linear Regression',
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'r2_score': model.score(X, y),
            'annual_growth_rate': annual_growth_rate
        }
    except Exception as e:
        st.error(f"Linear Regression Error: {str(e)}")
        return None

def prophet_forecast(data, date_col, target_col, periods, aggregation='Daily', annual_growth_rate=0.0, confidence_level=95):
    """Prophet forecasting model (simplified implementation)"""
    try:
        # Simplified Prophet-like implementation using polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        
        data_sorted = data.sort_values(date_col)
        X = np.arange(len(data_sorted)).reshape(-1, 1)
        y = data_sorted[target_col].values
        
        # Create polynomial features for seasonality
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate forecast
        future_X = np.arange(len(data_sorted), len(data_sorted) + periods).reshape(-1, 1)
        future_X_poly = poly_features.transform(future_X)
        forecast_values = model.predict(future_X_poly)
        
        # Calculate confidence intervals for Prophet-like model
        residuals = y - model.predict(X_poly)
        mse = np.mean(residuals**2)
        confidence_multiplier = get_confidence_multiplier(confidence_level)
        
        # Calculate prediction intervals (simplified approach)
        prediction_std = np.sqrt(mse) * np.sqrt(1 + 1/len(X_poly))
        
        # Calculate confidence bounds
        lower_bound = forecast_values - confidence_multiplier * prediction_std
        upper_bound = forecast_values + confidence_multiplier * prediction_std
        
        # Apply annual growth rate if specified
        if annual_growth_rate != 0.0:
            # Convert annual growth rate to period growth rate
            period_growth_rate = annual_growth_rate / 100.0
            
            # Calculate growth multiplier for each period
            if aggregation == "Daily":
                period_multiplier = (1 + period_growth_rate) ** (1/365)  # Daily growth
            elif aggregation == "Weekly":
                period_multiplier = (1 + period_growth_rate) ** (1/52)   # Weekly growth
            elif aggregation == "Monthly":
                period_multiplier = (1 + period_growth_rate) ** (1/12)   # Monthly growth
            elif aggregation == "Yearly":
                period_multiplier = (1 + period_growth_rate)             # Yearly growth
            else:
                period_multiplier = 1.0
            
            # Apply growth to forecast values and bounds
            for i in range(len(forecast_values)):
                growth_factor = period_multiplier ** (i + 1)
                forecast_values[i] *= growth_factor
                lower_bound[i] *= growth_factor
                upper_bound[i] *= growth_factor
        
        # Create forecast dates
        last_date = data_sorted[date_col].iloc[-1]
        if hasattr(last_date, 'to_timestamp'):
            last_date = last_date.to_timestamp()
        elif isinstance(last_date, pd.Period):
            last_date = last_date.to_timestamp()
        
        # Determine frequency based on aggregation
        freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME', 'Yearly': 'YE'}
        freq = freq_map.get(aggregation, 'D')
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq=freq)
        
        return {
            'forecast_values': forecast_values,
            'forecast_dates': forecast_dates,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'model_name': 'Prophet (Polynomial)',
            'slope': model.coef_[0] if len(model.coef_) > 0 else 0,
            'intercept': model.intercept_,
            'r2_score': model.score(X_poly, y),
            'annual_growth_rate': annual_growth_rate
        }
    except Exception as e:
        st.error(f"Prophet Error: {str(e)}")
        return None

def ensemble_forecast(data, date_col, target_col, periods, aggregation='Daily', annual_growth_rate=0.0, confidence_level=95):
    """Ensemble forecasting model (ARIMA + Linear Trend)"""
    try:
        data_sorted = data.sort_values(date_col)
        X = np.arange(len(data_sorted)).reshape(-1, 1)
        y = data_sorted[target_col].values
        
        # Simple ARIMA-like implementation using moving averages
        window_size = min(7, len(y) // 4)  # Adaptive window size
        if window_size < 2:
            window_size = 2
            
        # Calculate moving average trend
        ma_trend = pd.Series(y).rolling(window=window_size, min_periods=1).mean().values
        
        # Linear trend component
        linear_model = LinearRegression()
        linear_model.fit(X, ma_trend.reshape(-1, 1))
        
        # Generate forecast combining both components
        future_X = np.arange(len(data_sorted), len(data_sorted) + periods).reshape(-1, 1)
        
        # Linear trend forecast
        linear_forecast = linear_model.predict(future_X)
        
        # Add some noise for realism
        noise = np.random.normal(0, np.std(y) * 0.1, periods)
        forecast_values = linear_forecast.flatten() + noise
        
        # Calculate confidence intervals for ensemble model
        confidence_multiplier = get_confidence_multiplier(confidence_level)
        
        # Calculate prediction uncertainty
        ensemble_std = np.std(y) * 0.15  # Ensemble uncertainty
        prediction_std = np.full(periods, ensemble_std)
        
        # Calculate confidence bounds
        lower_bound = forecast_values - confidence_multiplier * prediction_std
        upper_bound = forecast_values + confidence_multiplier * prediction_std
        
        # Apply annual growth rate if specified
        if annual_growth_rate != 0.0:
            # Convert annual growth rate to period growth rate
            period_growth_rate = annual_growth_rate / 100.0
            
            # Calculate growth multiplier for each period
            if aggregation == "Daily":
                period_multiplier = (1 + period_growth_rate) ** (1/365)  # Daily growth
            elif aggregation == "Weekly":
                period_multiplier = (1 + period_growth_rate) ** (1/52)   # Weekly growth
            elif aggregation == "Monthly":
                period_multiplier = (1 + period_growth_rate) ** (1/12)   # Monthly growth
            elif aggregation == "Yearly":
                period_multiplier = (1 + period_growth_rate)             # Yearly growth
            else:
                period_multiplier = 1.0
            
            # Apply growth to forecast values and bounds
            for i in range(len(forecast_values)):
                growth_factor = period_multiplier ** (i + 1)
                forecast_values[i] *= growth_factor
                lower_bound[i] *= growth_factor
                upper_bound[i] *= growth_factor
        
        # Create forecast dates
        last_date = data_sorted[date_col].iloc[-1]
        if hasattr(last_date, 'to_timestamp'):
            last_date = last_date.to_timestamp()
        elif isinstance(last_date, pd.Period):
            last_date = last_date.to_timestamp()
        
        # Determine frequency based on aggregation
        freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME', 'Yearly': 'YE'}
        freq = freq_map.get(aggregation, 'D')
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq=freq)
        
        return {
            'forecast_values': forecast_values,
            'forecast_dates': forecast_dates,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'model_name': 'Ensemble (ARIMA + Linear)',
            'slope': linear_model.coef_[0][0],
            'intercept': linear_model.intercept_[0],
            'r2_score': linear_model.score(X, ma_trend.reshape(-1, 1)),
            'annual_growth_rate': annual_growth_rate
        }
    except Exception as e:
        st.error(f"Ensemble Error: {str(e)}")
        return None

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">📈 AI Sales Forecasting Demo</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">Advanced forecasting for food manufacturing with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

def load_sample_data_demo():
    """Load realistic sample data for demo"""
    if st.button("📊 Load Sample Data", type="primary", width='stretch'):
        with st.spinner("Generating realistic sample data..."):
            # Realistic product and customer names
            products = [
                ('FF0001', 'Frozen Peas'), ('FF0002', 'Frozen Corn'), ('FF0003', 'Frozen Carrots'),
                ('FF0004', 'Frozen Spinach'), ('FF0005', 'Frozen Broccoli'), ('FF0006', 'Frozen Pizza'),
                ('FF0007', 'Frozen Chicken Nuggets'), ('FF0008', 'Frozen Fish Fillets'), ('FF0009', 'Frozen French Fries'),
                ('FF0010', 'Frozen Ice Cream'), ('FF0011', 'Frozen Waffles'), ('FF0012', 'Frozen Meatballs'),
                ('FF0013', 'Frozen Shrimp'), ('FF0014', 'Frozen Lasagna'), ('FF0015', 'Frozen Cauliflower'),
                ('FF0016', 'Frozen Green Beans'), ('FF0017', 'Frozen Mixed Vegetables'), ('FF0018', 'Frozen Berries'),
                ('FF0019', 'Frozen Yogurt'), ('FF0020', 'Frozen Bread')
            ]
            
            customers = [
                'FreshMart Grocery', 'HealthFoods Market', 'SuperValue Stores', 'Gourmet Foods Co',
                'QuickStop Convenience', 'Farmers Market Chain', 'Organic Plus', 'Budget Foods',
                'Premium Grocers', 'City Market', 'Village Foods', 'Metro Supermarket',
                'Green Grocery Co', 'Family Foods', 'Urban Market'
            ]
            
            data_rows = []
            start_date = datetime(2024, 1, 1)
            
            for i in range(3000):  # Generate 3000 rows
                # Select random product
                product_idx = np.random.randint(0, len(products))
                item_id, description = products[product_idx]
                
                # Select random customer
                customer = np.random.choice(customers)
                
                # Generate date (monthly data)
                month_offset = i // 250  # ~250 records per month
                current_date = start_date + timedelta(days=month_offset * 30 + np.random.randint(0, 30))
                
                # Generate sales with seasonal patterns
                base_sales = np.random.uniform(100, 1000)
                
                # Seasonal adjustments
                month = current_date.month
                if 'Ice Cream' in description or 'Frozen Yogurt' in description:
                    # Higher sales in summer
                    if month in [6, 7, 8]:
                        base_sales *= np.random.uniform(1.5, 2.5)
                    elif month in [12, 1, 2]:
                        base_sales *= np.random.uniform(0.3, 0.7)
                elif 'Pizza' in description or 'Lasagna' in description:
                    # Higher sales in winter/holidays
                    if month in [11, 12, 1]:
                        base_sales *= np.random.uniform(1.3, 1.8)
                elif 'Vegetables' in description or 'Broccoli' in description or 'Spinach' in description:
                    # Steady year-round with slight winter increase
                    if month in [11, 12, 1, 2]:
                        base_sales *= np.random.uniform(1.1, 1.3)
                
                # Add trend type
                trend_type = np.random.choice(['Regular', 'Irregular'], p=[0.75, 0.25])
                if trend_type == 'Irregular':
                    # Add more volatility
                    base_sales *= np.random.uniform(0.5, 1.5)
                
                sales_value = max(0, base_sales + np.random.normal(0, base_sales * 0.1))
                
                data_rows.append({
                    'ItemID': item_id,
                    'Description': description,
                    'TrendType': trend_type,
                    'ds': current_date.strftime('%Y-%m-%d'),
                    'CustomerName': customer,
                    'y': round(sales_value, 2)
                })
            
            df = pd.DataFrame(data_rows)
            
            st.session_state.uploaded_data = df
            st.session_state.current_step = 'validation'
            st.session_state.demo_mode = True
            st.success("✅ Sample data loaded successfully!")
            st.rerun()

def render_file_upload():
    """Render file upload section"""
    st.markdown('<h2 class="section-header">📁 Upload Your Data</h2>', unsafe_allow_html=True)
    
    # Sample data button
    load_sample_data_demo()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files with sales data"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Save file to user's directory
            if st.session_state.user_data_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_extension = uploaded_file.name.split('.')[-1]
                saved_filename = f"upload_{timestamp}.{file_extension}"
                save_path = os.path.join(st.session_state.user_data_path, saved_filename)
                
                if file_extension == 'csv':
                    df.to_csv(save_path, index=False)
                else:
                    df.to_excel(save_path, index=False)
                
                st.session_state.uploaded_file_path = save_path
            
            st.session_state.uploaded_data = df
            st.session_state.current_step = 'validation'
            st.session_state.demo_mode = False
            st.success("✅ File uploaded successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def render_data_validation():
    """Render data validation section"""
    st.markdown('<h2 class="section-header">✅ Data Validation</h2>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        
        # Data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), width='stretch')
        
        # Data summary
        st.subheader("📈 Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Auto-detect columns using OpenAI
        st.subheader("🤖 AI Column Detection")
        if st.button("🔍 Auto-Detect Columns", type="primary"):
            with st.spinner("AI is analyzing your data..."):
                # Prepare data sample for AI analysis
                sample_data = df.head(5).to_string()
                column_names = list(df.columns)
                
                # Create AI prompt
                prompt = f"""
                Analyze this sales data sample and identify the correct column mappings:
                
                Data Sample:
                {sample_data}
                
                Column Names: {column_names}
                
                Please identify which columns should be mapped to:
                1. date_col: Date/time column
                2. target_col: Sales/quantity/value column  
                3. product_col: Product/item identifier column
                4. customer_col: Customer/client identifier column
                
                Return your response as JSON format:
                {{
                    "date_col": "column_name",
                    "target_col": "column_name", 
                    "product_col": "column_name",
                    "customer_col": "column_name",
                    "confidence": "high/medium/low",
                    "reasoning": "explanation of choices"
                }}
                """
                
                try:
                    # Check if OpenAI is configured
                    is_configured, message = setup_openai()
                    if is_configured:
                        api_key = os.getenv('OPENAI_API_KEY')
                        client = openai.OpenAI(api_key=api_key)
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a data analysis expert. Analyze the provided data sample and identify the correct column mappings for sales forecasting."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=300,
                            temperature=0.3
                        )
                        
                        ai_response = response.choices[0].message.content
                        
                        # Parse JSON response
                        try:
                            # Extract JSON from response
                            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                            if json_match:
                                ai_mapping = json.loads(json_match.group())
                                st.session_state.ai_detected_mapping = ai_mapping
                                
                                # Auto-populate column selections
                                st.session_state.selected_date_col = ai_mapping.get('date_col')
                                st.session_state.selected_target_col = ai_mapping.get('target_col')
                                st.session_state.selected_product_col = ai_mapping.get('product_col')
                                st.session_state.selected_customer_col = ai_mapping.get('customer_col')
                                
                                st.success("✅ AI column detection completed!")
                                st.rerun()
                            else:
                                st.error("❌ Could not parse AI response")
                        except json.JSONDecodeError:
                            st.error("❌ Invalid JSON response from AI")
                    else:
                        st.warning(f"⚠️ {message}")
                        
                except Exception as e:
                    st.error(f"❌ Error calling OpenAI: {str(e)}")
        
        # Manual column mapping
        st.subheader("📋 Column Mapping")
        col1, col2 = st.columns(2)
        
        with col1:
            ai_mapping = st.session_state.ai_detected_mapping or {}
            
            # Safe index calculation for date column
            date_col_index = 0
            if ai_mapping.get('date_col') and ai_mapping.get('date_col') in df.columns:
                date_col_index = df.columns.get_loc(ai_mapping.get('date_col'))
            elif len(df.columns) > 0:
                date_col_index = 0
            
            st.session_state.selected_date_col = st.selectbox(
                "Date Column",
                df.columns,
                index=date_col_index,
                key="date_col"
            )
            
            # Safe index calculation for target column
            target_col_index = len(df.columns) - 1 if len(df.columns) > 0 else 0
            if ai_mapping.get('target_col') and ai_mapping.get('target_col') in df.columns:
                target_col_index = df.columns.get_loc(ai_mapping.get('target_col'))
            elif len(df.columns) > 1:
                target_col_index = len(df.columns) - 1
            
            st.session_state.selected_target_col = st.selectbox(
                "Target Column (Sales/Quantity)",
                df.columns,
                index=target_col_index,
                key="target_col"
            )
        
        with col2:
            # Safe index calculation for product column
            product_col_index = 1 if len(df.columns) > 1 else 0
            if ai_mapping.get('product_col') and ai_mapping.get('product_col') in df.columns:
                product_col_index = df.columns.get_loc(ai_mapping.get('product_col'))
            elif len(df.columns) > 1:
                product_col_index = 1
            
            st.session_state.selected_product_col = st.selectbox(
                "Product Column",
                df.columns,
                index=product_col_index,
                key="product_col"
            )
            
            # Safe index calculation for customer column
            customer_col_index = 2 if len(df.columns) > 2 else (1 if len(df.columns) > 1 else 0)
            if ai_mapping.get('customer_col') and ai_mapping.get('customer_col') in df.columns:
                customer_col_index = df.columns.get_loc(ai_mapping.get('customer_col'))
            elif len(df.columns) > 2:
                customer_col_index = 2
            elif len(df.columns) > 1:
                customer_col_index = 1
            
            st.session_state.selected_customer_col = st.selectbox(
                "Customer Column",
                df.columns,
                index=customer_col_index,
                key="customer_col"
            )
        
        # Show AI detection results if available
        if st.session_state.ai_detected_mapping:
            ai_mapping = st.session_state.ai_detected_mapping
            confidence = ai_mapping.get('confidence', 'Unknown')
            reasoning = ai_mapping.get('reasoning', 'No reasoning provided')
            st.info(f"🤖 **AI Confidence:** {confidence.title()} | **Reasoning:** {reasoning}")
        
        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back to Upload", type="secondary", width='stretch'):
                st.session_state.current_step = 'upload'
                st.rerun()
        
        with col2:
            if st.button("🚀 Continue to Forecasting", type="primary", width='stretch', key="continue_btn"):
                st.session_state.current_step = 'forecasting'
                st.rerun()

def render_forecasting():
    """Render forecasting section"""
    st.markdown('<h2 class="section-header">🔮 Forecasting Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        
        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Forecast Settings")
            
            # OpenAI Status
            st.subheader("🤖 AI Status")
            is_openai_configured, openai_message = setup_openai()
            if is_openai_configured:
                st.success("✅ OpenAI Connected")
            else:
                st.warning(f"⚠️ {openai_message}")
                st.info("💡 To enable AI chat, add your OpenAI API key to the .env file")
            
            st.markdown("---")
            
            # Model selection
            st.subheader("🎯 Forecasting Model")
            model_type = st.selectbox(
                "Select Forecasting Model",
                ["Linear Regression", "Prophet", "Ensemble"],
                help="""📊 **Linear Regression**: Simple trend-based forecasting. Best for data with clear linear trends.
                
📈 **Prophet**: Advanced time series with seasonality. Ideal for data with seasonal patterns and holidays.

🔄 **Ensemble**: Combines multiple methods (ARIMA + Linear). Best for complex, irregular patterns."""
            )
            
            # Data aggregation
            st.subheader("📅 Data Aggregation")
            aggregation = st.selectbox(
                "Aggregate Data By",
                ["Daily", "Weekly", "Monthly", "Yearly"],
                help="""📅 **Daily**: Raw daily data. Shows day-to-day variations. Best for short-term forecasting.

📊 **Weekly**: Weekly aggregation. Smooths daily noise. Good for medium-term planning.

📈 **Monthly**: Monthly aggregation. Highlights seasonal trends. Ideal for business planning.

📋 **Yearly**: Annual aggregation. Shows long-term trends. Best for strategic forecasting."""
            )
            
            # Dynamic forecast period based on aggregation
            if aggregation == "Daily":
                periods = st.slider("Forecast Period (days)", 1, 90, 30)
                period_label = "days"
            elif aggregation == "Weekly":
                periods = st.slider("Forecast Period (weeks)", 1, 12, 4)
                period_label = "weeks"
            elif aggregation == "Monthly":
                periods = st.slider("Forecast Period (months)", 1, 12, 3)
                period_label = "months"
            elif aggregation == "Yearly":
                periods = st.slider("Forecast Period (years)", 1, 5, 2)
                period_label = "years"
            
            confidence_level = st.selectbox("Confidence Level", [80, 95, 99], index=1)
            
            # Add explanation for confidence levels
            with st.expander("ℹ️ What does Confidence Level mean?"):
                st.markdown("""
                **Confidence Level** represents the statistical certainty of your forecast:
                
                - **80% Confidence**: Narrower bands, less uncertainty, but 20% chance the actual values could be outside the range
                - **95% Confidence**: Standard statistical confidence, 5% chance the actual values could be outside the range  
                - **99% Confidence**: Wider bands, very high certainty, but only 1% chance the actual values could be outside the range
                
                **Tip**: Use 95% for most business decisions. Use 80% for aggressive planning or 99% for conservative risk management.
                """)
            
            # Annual growth assumption
            st.subheader("📈 Growth Assumptions")
            annual_growth_rate = st.slider(
                "Annual Growth Rate (%)", 
                min_value=-20.0, 
                max_value=50.0, 
                value=0.0, 
                step=0.5,
                help="Expected annual growth rate. Positive values assume growth, negative values assume decline. This will be applied to the forecast."
            )
            
            # Data filters
            st.subheader("🔍 Data Filters")
            products = st.multiselect("Select Products", df[st.session_state.selected_product_col].unique())
            customers = st.multiselect("Select Customers", df[st.session_state.selected_customer_col].unique())
            
            # Generate forecast button
            if st.button("🚀 Generate Forecast", type="primary", width='stretch'):
                with st.spinner("Generating forecast..."):
                    # Apply filters
                    data = df.copy()
                    if products:
                        data = data[data[st.session_state.selected_product_col].isin(products)]
                    if customers:
                        data = data[data[st.session_state.selected_customer_col].isin(customers)]
                    
                    if len(data) == 0:
                        st.error("❌ No data remaining after applying filters!")
                        return
                    
                    # Convert date column
                    data[st.session_state.selected_date_col] = pd.to_datetime(data[st.session_state.selected_date_col])
                    
                    # Aggregate data based on selection
                    aggregated_data = aggregate_data(data, st.session_state.selected_date_col, st.session_state.selected_target_col, aggregation)
                    
                    # Generate forecast based on selected model
                    if model_type == "Linear Regression":
                        forecast_result = linear_regression_forecast(aggregated_data, st.session_state.selected_date_col, st.session_state.selected_target_col, periods, aggregation, annual_growth_rate, confidence_level)
                    elif model_type == "Prophet":
                        forecast_result = prophet_forecast(aggregated_data, st.session_state.selected_date_col, st.session_state.selected_target_col, periods, aggregation, annual_growth_rate, confidence_level)
                    elif model_type == "Ensemble":
                        forecast_result = ensemble_forecast(aggregated_data, st.session_state.selected_date_col, st.session_state.selected_target_col, periods, aggregation, annual_growth_rate, confidence_level)
                    
                    if forecast_result:
                        # Store forecast data
                        st.session_state.forecast_data = {
                            'target_col': st.session_state.selected_target_col,
                            'forecast_values': forecast_result['forecast_values'],
                            'periods': periods,
                            'period_label': period_label,
                            'slope': forecast_result['slope'],
                            'intercept': forecast_result['intercept'],
                            'historical_data': aggregated_data.values.tolist(),
                            'forecast_dates': forecast_result['forecast_dates'],
                            'model_name': forecast_result['model_name'],
                            'r2_score': forecast_result['r2_score'],
                            'aggregation': aggregation,
                            'annual_growth_rate': annual_growth_rate,
                            'confidence_level': confidence_level,
                            'original_data': data
                        }
                        
                        st.success(f"✅ {model_type} forecast generated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate forecast. Please try again.")
        
        # Main content area
        if st.session_state.forecast_data:
            forecast_data = st.session_state.forecast_data
            
            # Historical aggregated data chart
            st.subheader("📈 Historical Aggregated Data")
            
            # Create historical data chart
            historical_df = pd.DataFrame(forecast_data['historical_data'], columns=[st.session_state.selected_date_col, st.session_state.selected_target_col])
            historical_df[st.session_state.selected_date_col] = pd.to_datetime(historical_df[st.session_state.selected_date_col])
            
            # Create Plotly chart for historical data
            fig_historical = go.Figure()
            
            fig_historical.add_trace(go.Scatter(
                x=historical_df[st.session_state.selected_date_col],
                y=historical_df[st.session_state.selected_target_col],
                mode='lines+markers',
                name=f'Historical {forecast_data["aggregation"]} Data',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))
            
            fig_historical.update_layout(
                title=f"Historical Data - {forecast_data['aggregation']} Aggregation",
                xaxis_title="Date",
                yaxis_title=f"{st.session_state.selected_target_col} ({forecast_data['aggregation']})",
                hovermode='x unified',
                width=800,
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_historical, width='stretch')
            
            # Historical data table and forecast chart side by side
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Historical Data Table")
                historical_display_df = historical_df.copy()
                historical_display_df.columns = ['Date', f'{st.session_state.selected_target_col} ({forecast_data["aggregation"]})']
                st.dataframe(historical_display_df, width='stretch', hide_index=True)
            
            with col2:
                st.subheader("📊 Forecast Visualization")
                
                # Create Plotly chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=historical_df[st.session_state.selected_date_col],
                    y=historical_df[st.session_state.selected_target_col],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast_dates'],
                    y=forecast_data['forecast_values'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Add confidence interval if available
                if 'lower_bound' in forecast_data and 'upper_bound' in forecast_data:
                    confidence_level = forecast_data.get('confidence_level', 95)
                    fig.add_trace(go.Scatter(
                        x=forecast_data['forecast_dates'].tolist() + forecast_data['forecast_dates'].tolist()[::-1],
                        y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
                        fill='tonexty',
                        fillcolor=f'rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{confidence_level}% Confidence Interval',
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                
                fig.update_layout(
                    title=f"{forecast_data['model_name']} Forecast - {forecast_data['aggregation']} Aggregation",
                    xaxis_title="Date",
                    yaxis_title=st.session_state.selected_target_col,
                    hovermode='x unified',
                    width=400,
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # Model comparison
            st.subheader("🔄 Model Comparison")
            if st.button("📊 Generate All Model Comparison", type="secondary"):
                with st.spinner("Generating model comparison..."):
                    # Get aggregated data
                    data = forecast_data['original_data'].copy()
                    aggregated_data = aggregate_data(data, st.session_state.selected_date_col, st.session_state.selected_target_col, forecast_data['aggregation'])
                    
                    # Generate forecasts for all models
                    models_results = {}
                    models_results['Linear Regression'] = linear_regression_forecast(aggregated_data, st.session_state.selected_date_col, st.session_state.selected_target_col, forecast_data['periods'], forecast_data['aggregation'], forecast_data.get('annual_growth_rate', 0.0), forecast_data.get('confidence_level', 95))
                    models_results['Prophet'] = prophet_forecast(aggregated_data, st.session_state.selected_date_col, st.session_state.selected_target_col, forecast_data['periods'], forecast_data['aggregation'], forecast_data.get('annual_growth_rate', 0.0), forecast_data.get('confidence_level', 95))
                    models_results['Ensemble'] = ensemble_forecast(aggregated_data, st.session_state.selected_date_col, st.session_state.selected_target_col, forecast_data['periods'], forecast_data['aggregation'], forecast_data.get('annual_growth_rate', 0.0), forecast_data.get('confidence_level', 95))
                    
                    # Create comparison chart
                    fig_compare = go.Figure()
                    
                    # Historical data
                    fig_compare.add_trace(go.Scatter(
                        x=historical_df[st.session_state.selected_date_col],
                        y=historical_df[st.session_state.selected_target_col],
                        mode='lines+markers',
                        name='Historical Data',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Model forecasts
                    colors = ['#ff7f0e', '#2ca02c', '#d62728']
                    for i, (model_name, result) in enumerate(models_results.items()):
                        if result:
                            fig_compare.add_trace(go.Scatter(
                                x=result['forecast_dates'],
                                y=result['forecast_values'],
                                mode='lines+markers',
                                name=f'{model_name} Forecast',
                                line=dict(color=colors[i], width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            # Add confidence intervals for model comparison
                            if 'lower_bound' in result and 'upper_bound' in result:
                                confidence_level = result.get('confidence_level', 95)
                                confidence_color = colors[i].replace('rgb', 'rgba').replace(')', ', 0.15)')
                                fig_compare.add_trace(go.Scatter(
                                    x=result['forecast_dates'].tolist() + result['forecast_dates'].tolist()[::-1],
                                    y=result['upper_bound'].tolist() + result['lower_bound'].tolist()[::-1],
                                    fill='tonexty',
                                    fillcolor=confidence_color,
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name=f'{model_name} Confidence',
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                    
                    fig_compare.update_layout(
                        title="Model Comparison - All Forecasting Methods",
                        xaxis_title="Date",
                        yaxis_title=st.session_state.selected_target_col,
                        hovermode='x unified',
                        width=800,
                        height=500
                    )
                    
                    st.plotly_chart(fig_compare, width='stretch')
                    
                    # Performance comparison table
                    st.subheader("📈 Model Performance Comparison")
                    performance_data = []
                    for model_name, result in models_results.items():
                        if result:
                            performance_data.append({
                                'Model': model_name,
                                'R² Score': f"{result['r2_score']:.3f}",
                                'Trend Slope': f"{result['slope']:.4f}",
                                'Average Forecast': f"{np.mean(result['forecast_values']):.2f}"
                            })
                    
                    if performance_data:
                        performance_df = pd.DataFrame(performance_data)
                        st.dataframe(performance_df, width='stretch', hide_index=True)
            
            # Forecast summary
            st.subheader("📋 Forecast Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Used", forecast_data['model_name'])
            with col2:
                st.metric("Aggregation", forecast_data['aggregation'])
            with col3:
                st.metric("R² Score", f"{forecast_data['r2_score']:.3f}")
            with col4:
                st.metric("Forecast Periods", f"{forecast_data['periods']} {forecast_data.get('period_label', 'days')}")
            
            # AI Forecast Narrative
            st.subheader("🤖 AI Forecast Narrative")
            if st.button("📝 Generate Forecast Narrative", type="secondary"):
                with st.spinner("AI is analyzing your forecast..."):
                    narrative_prompt = f"""
                    Provide a comprehensive business narrative about this sales forecast for food manufacturing:
                    
                    Forecast Details:
                    - Model: {forecast_data['model_name']}
                    - Aggregation: {forecast_data['aggregation']}
                    - Target Column: {forecast_data['target_col']}
                    - Forecast Period: {forecast_data['periods']} {forecast_data.get('period_label', 'days')}
                    - Trend Direction: {'Increasing' if forecast_data['slope'] > 0 else 'Decreasing'}
                    - Trend Strength: {abs(forecast_data['slope']):.4f}
                    - R² Score: {forecast_data['r2_score']:.3f}
                    - Average Forecast Value: {sum(forecast_data['forecast_values'])/len(forecast_data['forecast_values']):.2f}
                    - Forecast Range: {min(forecast_data['forecast_values']):.2f} to {max(forecast_data['forecast_values']):.2f}
                    - Historical Data Points: {len(forecast_data['historical_data'])}
                    
                    Please provide a professional business narrative that includes:
                    1. Executive summary of the forecast
                    2. Key insights and patterns
                    3. Business implications
                    4. Recommendations for action
                    5. Risk factors to consider
                    
                    Write in a clear, professional tone suitable for business stakeholders.
                    """
                    
                    try:
                        # Check if OpenAI is configured
                        is_configured, message = setup_openai()
                        if is_configured:
                            api_key = os.getenv('OPENAI_API_KEY')
                            client = openai.OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a business analyst and forecasting expert specializing in food manufacturing. Provide clear, actionable business insights about sales forecasts."},
                                    {"role": "user", "content": narrative_prompt}
                                ],
                                max_tokens=800,
                                temperature=0.7
                            )
                            
                            narrative = response.choices[0].message.content
                            
                            # Store narrative in session state
                            st.session_state.forecast_narrative = narrative
                            
                            st.success("✅ Forecast narrative generated!")
                        else:
                            st.warning(f"⚠️ {message}")
                            st.info("💡 To generate AI narratives, add your OpenAI API key to the .env file")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating narrative: {str(e)}")
            
            # Display narrative if available
            if 'forecast_narrative' in st.session_state and st.session_state.forecast_narrative:
                st.markdown("---")
                st.markdown("### 📊 **Forecast Analysis**")
                st.markdown(st.session_state.forecast_narrative)
                
                # Download narrative button
                narrative_text = st.session_state.forecast_narrative
                st.download_button(
                    label="📥 Download Narrative as Text",
                    data=narrative_text,
                    file_name=f"forecast_narrative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                trend_direction = "📈 Increasing" if forecast_data['slope'] > 0 else "📉 Decreasing"
                st.metric("Trend", trend_direction)
            with col2:
                growth_rate = forecast_data.get('annual_growth_rate', 0.0)
                growth_display = f"{growth_rate:+.1f}%" if growth_rate != 0.0 else "0.0%"
                st.metric("Annual Growth", growth_display)
            with col3:
                total_forecast = sum(forecast_data['forecast_values'])
                st.metric("Total Forecast", f"{total_forecast:.2f}")
            with col4:
                avg_forecast = sum(forecast_data['forecast_values']) / len(forecast_data['forecast_values'])
                st.metric("Average Forecast", f"{avg_forecast:.2f}")
            
            # Second row of metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Minimum Forecast", f"{min(forecast_data['forecast_values']):.2f}")
            with col2:
                st.metric("Maximum Forecast", f"{max(forecast_data['forecast_values']):.2f}")
            with col3:
                trend_strength = abs(forecast_data['slope'])
                st.metric("Trend Strength", f"{trend_strength:.4f}")
            with col4:
                # Calculate growth impact
                if growth_rate != 0.0:
                    first_forecast = forecast_data['forecast_values'][0]
                    last_forecast = forecast_data['forecast_values'][-1]
                    growth_impact = ((last_forecast - first_forecast) / first_forecast) * 100
                    st.metric("Growth Impact", f"{growth_impact:+.1f}%")
                else:
                    st.metric("Growth Impact", "0.0%")
            
            # Forecast statistics and data table side by side
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📈 Forecast Statistics")
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.metric("Minimum Forecast", f"{min(forecast_data['forecast_values']):.2f}")
                with col_stat2:
                    st.metric("Maximum Forecast", f"{max(forecast_data['forecast_values']):.2f}")
                
                total_forecast = sum(forecast_data['forecast_values'])
                st.metric("Total Forecast", f"{total_forecast:.2f}")
            
            with col2:
                st.subheader("📋 Forecast Data Table")
                forecast_df = pd.DataFrame({
                    'Date': forecast_data['forecast_dates'],
                    'Forecast': forecast_data['forecast_values']
                })
                st.dataframe(forecast_df, width='stretch', hide_index=True)
            
            # Download forecast as CSV
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"forecast_{st.session_state.selected_target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # AI Chat Section
            st.subheader("🤖 AI Chat - Ask Questions About Your Forecast")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_question = st.text_input(
                    "Ask a question about your forecast:",
                    placeholder="e.g., What does this trend mean for my business?",
                    key="ai_question_input"
                )
            
            with col2:
                if st.button("💬 Ask AI", type="primary"):
                    if user_question:
                        with st.spinner("AI is thinking..."):
                            ai_response = generate_ai_response(user_question, forecast_data)
                            
                            # Store in chat history
                            st.session_state.chat_history.append({
                                'question': user_question,
                                'answer': ai_response,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    else:
                        st.warning("Please enter a question first!")
            
            # Display chat history
            if st.session_state.chat_history:
                with st.expander("💬 Chat History", expanded=True):
                    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                        st.markdown(f"**Q:** {chat['question']}")
                        st.markdown(f"**A:** {chat['answer']}")
                        st.markdown(f"*{chat['timestamp']}*")
                        if i < len(st.session_state.chat_history[-5:]) - 1:
                            st.markdown("---")
            
            # Quick question buttons
            st.subheader("💡 Quick Questions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📈 What's the trend?", key="btn_trend"):
                    # Generate AI response directly
                    ai_response = generate_ai_response("What does the trend in this forecast mean for my business?", forecast_data)
                    st.markdown(f"""
                    **📈 Trend Analysis:**
                    
                    {ai_response}
                    """)
            
            with col2:
                if st.button("⚠️ Any risks?", key="btn_risks"):
                    # Generate AI response directly
                    ai_response = generate_ai_response("What are the potential risks or concerns with this forecast?", forecast_data)
                    st.markdown(f"""
                    **⚠️ Risk Assessment:**
                    
                    {ai_response}
                    """)
            
            with col3:
                if st.button("🎯 What should I focus on?", key="btn_focus"):
                    # Generate AI response directly
                    ai_response = generate_ai_response("What should I focus on based on this forecast?", forecast_data)
                    st.markdown(f"""
                    **🎯 Focus Areas:**
                    
                    {ai_response}
                    """)
            
            with col4:
                if st.button("📊 How accurate is this?", key="btn_accuracy"):
                    # Generate AI response directly
                    ai_response = generate_ai_response("How accurate is this forecast and what factors affect its reliability?", forecast_data)
                    st.markdown(f"""
                    **📊 Accuracy Assessment:**
                    
                    {ai_response}
                    """)
            
            # Navigation buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⬅️ Back to Validation", type="secondary", width='stretch'):
                    st.session_state.current_step = 'validation'
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset & Start Over", type="secondary", width='stretch'):
                    st.session_state.forecast_data = None
                    st.session_state.chat_history = []
                    st.session_state.current_step = 'upload'
                    st.rerun()
            
            with col3:
                if st.button("📁 Upload New Data", type="primary", width='stretch'):
                    st.session_state.uploaded_data = None
                    st.session_state.current_step = 'upload'
                    st.rerun()

def ensure_user_directory(username):
    """Create user-specific data directory if it doesn't exist"""
    user_dir = os.path.join('user_data', username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_user_role(username):
    """Get user role from config"""
    try:
        return config['credentials']['usernames'][username].get('role', 'customer')
    except:
        return 'customer'

def render_admin_panel():
    """Render admin panel for viewing all users' data"""
    st.markdown('<h2 class="section-header">🔧 Admin Panel</h2>', unsafe_allow_html=True)
    
    # Get list of all users
    user_data_dir = 'user_data'
    if os.path.exists(user_data_dir):
        users = [d for d in os.listdir(user_data_dir) if os.path.isdir(os.path.join(user_data_dir, d))]
        
        if users:
            st.subheader("View User Data")
            selected_user = st.selectbox("Select User to View", ['All Users'] + users)
            
            if selected_user != 'All Users':
                user_dir = os.path.join(user_data_dir, selected_user)
                user_files = [f for f in os.listdir(user_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
                
                if user_files:
                    st.write(f"**User:** {selected_user}")
                    st.write(f"**Files:** {len(user_files)}")
                    
                    for file in user_files:
                        with st.expander(f"📄 {file}"):
                            file_path = os.path.join(user_dir, file)
                            try:
                                if file.endswith('.csv'):
                                    df = pd.read_csv(file_path)
                                else:
                                    df = pd.read_excel(file_path)
                                st.dataframe(df.head(10))
                                st.write(f"Total rows: {len(df)}")
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
                else:
                    st.info(f"No data files found for user {selected_user}")
            else:
                # Show summary of all users
                st.write("### All Users Summary")
                for user in users:
                    user_dir = os.path.join(user_data_dir, user)
                    user_files = [f for f in os.listdir(user_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
                    st.write(f"**{user}**: {len(user_files)} file(s)")
        else:
            st.info("No user data found yet")
    else:
        st.info("No user data directory found yet")

def main():
    """Main application function with authentication"""
    # Check if user is already logged in
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.user_data_path = None
    
    # Show login form if not authenticated
    if not st.session_state.authenticated:
        st.title("🔐 Login to Sales Forecasting App")
        
        with st.form("login_form"):
            st.subheader("Please log in to continue")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if check_password(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = get_user_role(username)
                    st.session_state.user_data_path = ensure_user_directory(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.info("""
        **Demo Credentials:**
        
        **Customer Login:**
        - Username: `customer`
        - Password: `Customer123!`
        
        **Admin Login:**
        - Username: `admin`
        - Password: `Admin123!`
        """)
        return
    
    # User is authenticated - show main app
    initialize_session_state()
    
    # Add logout button in sidebar
    with st.sidebar:
        st.write(f"**Logged in as:** {get_user_name(st.session_state.username)}")
        st.write(f"**Role:** {st.session_state.role.title()}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.user_data_path = None
            st.rerun()
        st.markdown("---")
        
        # Admin panel link
        if st.session_state.role == 'admin':
            if st.button("🔧 Admin Panel", use_container_width=True):
                st.session_state.current_step = 'admin'
                st.rerun()
    
    # Main app content
    if st.session_state.current_step == 'admin' and st.session_state.role == 'admin':
        render_header()
        render_admin_panel()
    else:
        render_header()
        if st.session_state.current_step == 'upload':
            render_file_upload()
        elif st.session_state.current_step == 'validation':
            render_data_validation()
        elif st.session_state.current_step == 'forecasting':
            render_forecasting()

if __name__ == "__main__":
    main()