# Forecasting Module
# Comprehensive forecasting models for sales forecasting Streamlit app

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetForecaster:
    """Facebook Prophet forecasting model with seasonality handling"""
    
    def __init__(self, seasonality_mode: str = 'additive', 
                 yearly_seasonality: bool = True, 
                 weekly_seasonality: bool = False):
        """
        Initialize Prophet forecaster
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Enable yearly seasonality
            weekly_seasonality: Enable weekly seasonality
        """
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model = None
        self.fitted_data = None
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit Prophet model to data
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
            
        Returns:
            Dictionary with fit results
        """
        try:
            # Validate input data
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must contain 'ds' and 'y' columns")
            
            if len(df) < 2:
                raise ValueError("Need at least 2 data points for forecasting")
            
            # Initialize Prophet model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality
            )
            
            # Fit the model
            self.model.fit(df)
            self.fitted_data = df.copy()
            self.is_fitted = True
            
            logger.info(f"Prophet model fitted successfully on {len(df)} data points")
            
            return {
                'status': 'success',
                'message': 'Prophet model fitted successfully',
                'data_points': len(df),
                'seasonality_mode': self.seasonality_mode
            }
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate predictions
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Extract forecast data
            forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            forecast_data.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
            
            logger.info(f"Generated {periods} period forecast")
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error generating Prophet predictions: {str(e)}")
            raise
    
    def get_components(self) -> Dict[str, pd.DataFrame]:
        """
        Get trend and seasonal components
        
        Returns:
            Dictionary with component DataFrames
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting components")
        
        try:
            # Get components from the last prediction
            future = self.model.make_future_dataframe(periods=0)
            forecast = self.model.predict(future)
            
            components = {
                'trend': forecast[['ds', 'trend']].copy(),
                'yearly': forecast[['ds', 'yearly']].copy() if 'yearly' in forecast.columns else None,
                'weekly': forecast[['ds', 'weekly']].copy() if 'weekly' in forecast.columns else None
            }
            
            return components
            
        except Exception as e:
            logger.error(f"Error getting Prophet components: {str(e)}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and parameters
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_type': 'Prophet',
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'is_fitted': self.is_fitted,
            'data_points': len(self.fitted_data) if self.fitted_data is not None else 0
        }

class EnsembleForecaster:
    """Ensemble forecaster combining ARIMA and linear trend"""
    
    def __init__(self, arima_weight: float = 0.7, trend_weight: float = 0.3):
        """
        Initialize ensemble forecaster
        
        Args:
            arima_weight: Weight for ARIMA component
            trend_weight: Weight for linear trend component
        """
        self.arima_weight = arima_weight
        self.trend_weight = trend_weight
        self.arima_model = None
        self.trend_model = None
        self.fitted_data = None
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit ensemble model to data
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
            
        Returns:
            Dictionary with fit results
        """
        try:
            # Validate input data
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must contain 'ds' and 'y' columns")
            
            if len(df) < 10:  # Need more data for ARIMA
                raise ValueError("Need at least 10 data points for ensemble forecasting")
            
            # Prepare data
            df_sorted = df.sort_values('ds').reset_index(drop=True)
            y_values = df_sorted['y'].values
            
            # Fit ARIMA model
            try:
                self.arima_model = ARIMA(y_values, order=(1, 1, 1))
                self.arima_model = self.arima_model.fit()
            except:
                # Fallback to simpler ARIMA
                self.arima_model = ARIMA(y_values, order=(1, 0, 0))
                self.arima_model = self.arima_model.fit()
            
            # Fit linear trend model
            X_trend = np.arange(len(y_values)).reshape(-1, 1)
            self.trend_model = LinearRegression()
            self.trend_model.fit(X_trend, y_values)
            
            self.fitted_data = df_sorted.copy()
            self.is_fitted = True
            
            logger.info(f"Ensemble model fitted successfully on {len(df)} data points")
            
            return {
                'status': 'success',
                'message': 'Ensemble model fitted successfully',
                'data_points': len(df),
                'arima_weight': self.arima_weight,
                'trend_weight': self.trend_weight
            }
            
        except Exception as e:
            logger.error(f"Error fitting ensemble model: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate ensemble predictions
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Generate ARIMA predictions
            arima_forecast = self.arima_model.forecast(steps=periods)
            arima_conf_int = self.arima_model.get_forecast(steps=periods).conf_int()
            
            # Generate trend predictions
            last_index = len(self.fitted_data)
            trend_X = np.arange(last_index, last_index + periods).reshape(-1, 1)
            trend_forecast = self.trend_model.predict(trend_X)
            
            # Combine predictions
            combined_forecast = (self.arima_weight * arima_forecast + 
                               self.trend_weight * trend_forecast)
            
            # Calculate confidence intervals
            arima_std = (arima_conf_int.iloc[:, 1] - arima_conf_int.iloc[:, 0]) / 4
            trend_std = np.std(self.fitted_data['y']) * 0.1  # Simple approximation
            combined_std = np.sqrt((self.arima_weight * arima_std) ** 2 + 
                                 (self.trend_weight * trend_std) ** 2)
            
            # Create future dates
            last_date = self.fitted_data['ds'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=periods, freq='D')
            
            # Create result DataFrame
            forecast_data = pd.DataFrame({
                'date': future_dates,
                'forecast': combined_forecast,
                'lower_bound': combined_forecast - 1.96 * combined_std,
                'upper_bound': combined_forecast + 1.96 * combined_std
            })
            
            logger.info(f"Generated {periods} period ensemble forecast")
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'Ensemble',
            'arima_weight': self.arima_weight,
            'trend_weight': self.trend_weight,
            'is_fitted': self.is_fitted,
            'data_points': len(self.fitted_data) if self.fitted_data is not None else 0
        }

class LinearRegressionForecaster:
    """Linear regression forecaster with time-based features"""
    
    def __init__(self, include_seasonality: bool = True, polynomial_degree: int = 1):
        """
        Initialize linear regression forecaster
        
        Args:
            include_seasonality: Include seasonal features
            polynomial_degree: Degree of polynomial features
        """
        self.include_seasonality = include_seasonality
        self.polynomial_degree = polynomial_degree
        self.model = None
        self.poly_features = None
        self.fitted_data = None
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit linear regression model to data
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
            
        Returns:
            Dictionary with fit results
        """
        try:
            # Validate input data
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must contain 'ds' and 'y' columns")
            
            if len(df) < 5:
                raise ValueError("Need at least 5 data points for linear regression")
            
            # Prepare features
            df_sorted = df.sort_values('ds').reset_index(drop=True)
            X = self._create_features(df_sorted)
            y = df_sorted['y'].values
            
            # Apply polynomial features if degree > 1
            if self.polynomial_degree > 1:
                self.poly_features = PolynomialFeatures(degree=self.polynomial_degree)
                X = self.poly_features.fit_transform(X)
            
            # Fit model
            self.model = LinearRegression()
            self.model.fit(X, y)
            
            self.fitted_data = df_sorted.copy()
            self.is_fitted = True
            
            logger.info(f"Linear regression model fitted successfully on {len(df)} data points")
            
            return {
                'status': 'success',
                'message': 'Linear regression model fitted successfully',
                'data_points': len(df),
                'include_seasonality': self.include_seasonality,
                'polynomial_degree': self.polynomial_degree
            }
            
        except Exception as e:
            logger.error(f"Error fitting linear regression model: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create time-based features"""
        features = []
        
        # Time index
        time_index = np.arange(len(df)).reshape(-1, 1)
        features.append(time_index)
        
        if self.include_seasonality:
            # Day of year
            day_of_year = df['ds'].dt.dayofyear.values.reshape(-1, 1)
            features.append(day_of_year)
            
            # Month
            month = df['ds'].dt.month.values.reshape(-1, 1)
            features.append(month)
            
            # Day of week
            day_of_week = df['ds'].dt.dayofweek.values.reshape(-1, 1)
            features.append(day_of_week)
        
        return np.hstack(features)
    
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate linear regression predictions
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Create future dates
            last_date = self.fitted_data['ds'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=periods, freq='D')
            
            # Create future DataFrame for feature creation
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Create features for future data
            X_future = self._create_features(future_df)
            
            # Apply polynomial features if needed
            if self.poly_features is not None:
                X_future = self.poly_features.transform(X_future)
            
            # Make predictions
            predictions = self.model.predict(X_future)
            
            # Calculate confidence intervals (simple approximation)
            residuals = self.fitted_data['y'] - self.model.predict(
                self._create_features(self.fitted_data) if self.poly_features is None 
                else self.poly_features.transform(self._create_features(self.fitted_data))
            )
            std_error = np.std(residuals)
            
            # Create result DataFrame
            forecast_data = pd.DataFrame({
                'date': future_dates,
                'forecast': predictions,
                'lower_bound': predictions - 1.96 * std_error,
                'upper_bound': predictions + 1.96 * std_error
            })
            
            logger.info(f"Generated {periods} period linear regression forecast")
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error generating linear regression predictions: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'Linear Regression',
            'include_seasonality': self.include_seasonality,
            'polynomial_degree': self.polynomial_degree,
            'is_fitted': self.is_fitted,
            'data_points': len(self.fitted_data) if self.fitted_data is not None else 0
        }

def generate_forecast(data: pd.DataFrame, model_type: str, 
                     model_params: Dict[str, Any], periods: int) -> Dict[str, Any]:
    """
    Main orchestrator function for generating forecasts
    
    Args:
        data: Input data with 'ds' and 'y' columns
        model_type: Type of model ('prophet', 'ensemble', 'linear')
        model_params: Model parameters
        periods: Number of periods to forecast
        
    Returns:
        Dictionary with forecast results
    """
    try:
        # Initialize model based on type
        if model_type.lower() == 'prophet':
            model = ProphetForecaster(**model_params)
        elif model_type.lower() == 'ensemble':
            model = EnsembleForecaster(**model_params)
        elif model_type.lower() == 'linear':
            model = LinearRegressionForecaster(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Fit model
        fit_result = model.fit(data)
        if fit_result['status'] != 'success':
            return fit_result
        
        # Generate predictions
        forecast_data = model.predict(periods)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(data, forecast_data)
        
        return {
            'status': 'success',
            'forecast_data': forecast_data,
            'model_info': model.get_model_info(),
            'metrics': metrics,
            'fit_result': fit_result
        }
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

def create_forecast_chart(historical_data: pd.DataFrame, 
                         forecast_data: pd.DataFrame,
                         title: str = "Sales Forecast",
                         model_name: str = None,
                         color_scheme: Dict[str, str] = None) -> go.Figure:
    """
    Create enhanced Plotly visualization for historical and forecast data
    
    Args:
        historical_data: Historical data with 'ds' and 'y' columns
        forecast_data: Forecast data with 'date', 'forecast', 'lower_bound', 'upper_bound'
        title: Chart title
        model_name: Name of the forecasting model
        color_scheme: Custom color scheme dictionary
        
    Returns:
        Plotly figure object
    """
    try:
        # Default professional color scheme
        if color_scheme is None:
            color_scheme = {
                'historical': '#1f77b4',  # Primary blue
                'forecast': '#e74c3c',    # Red
                'confidence': 'rgba(231, 76, 60, 0.2)',  # Light red
                'background': '#ffffff',
                'grid': '#ecf0f1',
                'text': '#2c3e50'
            }
        
        fig = go.Figure()
        
        # Add historical data with enhanced styling
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(
                color=color_scheme['historical'],
                width=3,
                shape='spline',
                smoothing=0.3
            ),
            marker=dict(
                size=6,
                color=color_scheme['historical'],
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Historical</b><br>' +
                         'Date: %{x}<br>' +
                         'Sales: $%{y:,.0f}<br>' +
                         '<extra></extra>',
            showlegend=True
        ))
        
        # Add forecast data with dashed line
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['forecast'],
            mode='lines+markers',
            name=f'Forecast ({model_name})' if model_name else 'Forecast',
            line=dict(
                color=color_scheme['forecast'],
                width=3,
                dash='dash',
                shape='spline',
                smoothing=0.3
            ),
            marker=dict(
                size=6,
                color=color_scheme['forecast'],
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Forecast</b><br>' +
                         'Date: %{x}<br>' +
                         'Predicted Sales: $%{y:,.0f}<br>' +
                         '<extra></extra>',
            showlegend=True
        ))
        
        # Add confidence interval as shaded area
        fig.add_trace(go.Scatter(
            x=forecast_data['date'].tolist() + forecast_data['date'].tolist()[::-1],
            y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
            fill='tonexty',
            fillcolor=color_scheme['confidence'],
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add annotations for key insights
        add_chart_annotations(fig, historical_data, forecast_data)
        
        # Enhanced layout with professional styling
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color=color_scheme['text']),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14, color=color_scheme['text']),
                tickfont=dict(size=12, color=color_scheme['text']),
                gridcolor=color_scheme['grid'],
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title='Sales ($)',
                titlefont=dict(size=14, color=color_scheme['text']),
                tickfont=dict(size=12, color=color_scheme['text']),
                gridcolor=color_scheme['grid'],
                showgrid=True,
                zeroline=False,
                tickformat='$,.0f'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=500,
            margin=dict(l=60, r=60, t=80, b=60),
            plot_bgcolor=color_scheme['background'],
            paper_bgcolor=color_scheme['background'],
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=12, color=color_scheme['text'])
            ),
            annotations=[
                dict(
                    text=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.02,
                    y=0.02,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(size=10, color='#7f8c8d')
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating forecast chart: {str(e)}")
        # Return empty figure on error
        return go.Figure()

def add_chart_annotations(fig: go.Figure, historical_data: pd.DataFrame, forecast_data: pd.DataFrame):
    """Add key insights as annotations to the chart"""
    try:
        # Find peak historical value
        max_historical_idx = historical_data['y'].idxmax()
        max_historical_date = historical_data.loc[max_historical_idx, 'ds']
        max_historical_value = historical_data.loc[max_historical_idx, 'y']
        
        # Find peak forecast value
        max_forecast_idx = forecast_data['forecast'].idxmax()
        max_forecast_date = forecast_data.loc[max_forecast_idx, 'date']
        max_forecast_value = forecast_data.loc[max_forecast_idx, 'forecast']
        
        # Add peak historical annotation
        fig.add_annotation(
            x=max_historical_date,
            y=max_historical_value,
            text=f"Peak Historical<br>${max_historical_value:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#1f77b4',
            ax=0,
            ay=-40,
            font=dict(size=10, color='#1f77b4'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#1f77b4',
            borderwidth=1
        )
        
        # Add peak forecast annotation
        fig.add_annotation(
            x=max_forecast_date,
            y=max_forecast_value,
            text=f"Peak Forecast<br>${max_forecast_value:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#e74c3c',
            ax=0,
            ay=40,
            font=dict(size=10, color='#e74c3c'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e74c3c',
            borderwidth=1
        )
        
    except Exception as e:
        logger.error(f"Error adding chart annotations: {str(e)}")

def create_model_comparison_chart(all_model_results: Dict[str, Any], 
                                historical_data: pd.DataFrame,
                                title: str = "Model Comparison") -> go.Figure:
    """
    Create comparison chart for multiple forecasting models
    
    Args:
        all_model_results: Dictionary with model results
        historical_data: Historical data with 'ds' and 'y' columns
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        # Professional color palette for different models
        model_colors = {
            'Prophet': '#1f77b4',      # Blue
            'Ensemble': '#ff7f0e',     # Orange
            'Linear Regression': '#2ca02c',  # Green
            'ARIMA': '#d62728',        # Red
            'Exponential Smoothing': '#9467bd'  # Purple
        }
        
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(
                color='#2c3e50',
                width=4,
                shape='spline',
                smoothing=0.3
            ),
            marker=dict(
                size=8,
                color='#2c3e50',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>Historical</b><br>' +
                         'Date: %{x}<br>' +
                         'Sales: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add forecast data for each model
        for model_name, result in all_model_results.items():
            if result and 'forecast_data' in result:
                forecast_data = result['forecast_data']
                color = model_colors.get(model_name, '#7f8c8d')
                
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['forecast'],
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(
                        color=color,
                        width=3,
                        dash='dash',
                        shape='spline',
                        smoothing=0.3
                    ),
                    marker=dict(
                        size=6,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'<b>{model_name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Predicted Sales: $%{y:,.0f}<br>' +
                                 '<extra></extra>'
                ))
                
                # Add confidence interval (lighter shade)
                confidence_color = color.replace('rgb', 'rgba').replace(')', ', 0.2)')
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'].tolist() + forecast_data['date'].tolist()[::-1],
                    y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
                    fill='tonexty',
                    fillcolor=confidence_color,
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_name} Confidence',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14, color='#2c3e50'),
                tickfont=dict(size=12, color='#2c3e50'),
                gridcolor='#ecf0f1',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title='Sales ($)',
                titlefont=dict(size=14, color='#2c3e50'),
                tickfont=dict(size=12, color='#2c3e50'),
                gridcolor='#ecf0f1',
                showgrid=True,
                zeroline=False,
                tickformat='$,.0f'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(l=60, r=60, t=100, b=60),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=11, color='#2c3e50')
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating model comparison chart: {str(e)}")
        return go.Figure()

def create_metrics_cards(forecast_results: Dict[str, Any], 
                        historical_data: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Create key forecast insights as metrics cards
    
    Args:
        forecast_results: Forecast results dictionary
        historical_data: Historical data for comparison
        
    Returns:
        Dictionary with metric values for Streamlit display
    """
    try:
        metrics = {}
        
        if not forecast_results or 'metrics' not in forecast_results:
            return metrics
        
        forecast_metrics = forecast_results['metrics']
        forecast_data = forecast_results.get('forecast_data', pd.DataFrame())
        
        # Basic forecast metrics
        metrics['total_forecast'] = {
            'label': 'Total Forecast',
            'value': f"${forecast_metrics['forecast_summary']['total_forecast']:,.0f}",
            'delta': None,
            'help': 'Total predicted sales over forecast period'
        }
        
        metrics['daily_average'] = {
            'label': 'Daily Average',
            'value': f"${forecast_metrics['forecast_summary']['average_daily']:,.0f}",
            'delta': None,
            'help': 'Average daily sales prediction'
        }
        
        # Growth rate with delta
        growth_rate = forecast_metrics['forecast_summary']['growth_rate_percent']
        metrics['growth_rate'] = {
            'label': 'Growth Rate',
            'value': f"{growth_rate:.1f}%",
            'delta': f"{growth_rate:+.1f}%" if growth_rate != 0 else None,
            'help': 'Forecasted growth compared to historical average'
        }
        
        # Confidence level
        confidence_level = forecast_metrics['forecast_summary']['confidence_level']
        confidence_color = {
            'high': '#27ae60',    # Green
            'medium': '#f39c12',  # Orange
            'low': '#e74c3c'      # Red
        }.get(confidence_level, '#7f8c8d')
        
        metrics['confidence'] = {
            'label': 'Confidence Level',
            'value': confidence_level.title(),
            'delta': None,
            'help': 'Model confidence in predictions',
            'color': confidence_color
        }
        
        # Peak period analysis
        if not forecast_data.empty:
            peak_date = forecast_data.loc[forecast_data['forecast'].idxmax(), 'date']
            peak_value = forecast_data['forecast'].max()
            
            metrics['peak_period'] = {
                'label': 'Peak Period',
                'value': peak_date.strftime('%Y-%m-%d'),
                'delta': f"${peak_value:,.0f}",
                'help': 'Date with highest predicted sales'
            }
        
        # Trend direction
        trend_direction = forecast_metrics['forecast_summary']['trend_direction']
        trend_color = '#27ae60' if trend_direction == 'increasing' else '#e74c3c' if trend_direction == 'decreasing' else '#7f8c8d'
        
        metrics['trend'] = {
            'label': 'Trend Direction',
            'value': trend_direction.title(),
            'delta': None,
            'help': 'Overall trend direction',
            'color': trend_color
        }
        
        # Model performance indicators
        model_info = forecast_results.get('model_info', {})
        model_type = model_info.get('model_type', 'Unknown')
        
        metrics['model_type'] = {
            'label': 'Model Used',
            'value': model_type,
            'delta': None,
            'help': 'Forecasting algorithm used'
        }
        
        # Data quality indicator
        if historical_data is not None:
            data_points = len(historical_data)
            quality_level = 'High' if data_points > 100 else 'Medium' if data_points > 30 else 'Low'
            quality_color = '#27ae60' if quality_level == 'High' else '#f39c12' if quality_level == 'Medium' else '#e74c3c'
            
            metrics['data_quality'] = {
                'label': 'Data Quality',
                'value': quality_level,
                'delta': f"{data_points} points",
                'help': 'Quality based on historical data points',
                'color': quality_color
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error creating metrics cards: {str(e)}")
        return {}

def create_accuracy_metrics_table(all_model_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create accuracy metrics table for model comparison
    
    Args:
        all_model_results: Dictionary with all model results
        
    Returns:
        DataFrame with accuracy metrics
    """
    try:
        metrics_data = []
        
        for model_name, result in all_model_results.items():
            if result and 'metrics' in result:
                metrics = result['metrics']
                forecast_summary = metrics.get('forecast_summary', {})
                confidence_metrics = metrics.get('confidence_metrics', {})
                
                metrics_data.append({
                    'Model': model_name,
                    'Total Forecast': f"${forecast_summary.get('total_forecast', 0):,.0f}",
                    'Daily Average': f"${forecast_summary.get('average_daily', 0):,.0f}",
                    'Growth Rate': f"{forecast_summary.get('growth_rate_percent', 0):.1f}%",
                    'Confidence': forecast_summary.get('confidence_level', 'Unknown').title(),
                    'Trend': forecast_summary.get('trend_direction', 'Unknown').title(),
                    'Avg Confidence Width': f"${confidence_metrics.get('avg_confidence_width', 0):,.0f}"
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error creating accuracy metrics table: {str(e)}")
        return pd.DataFrame()

def create_seasonality_chart(historical_data: pd.DataFrame, 
                           forecast_data: pd.DataFrame = None) -> go.Figure:
    """
    Create seasonality analysis chart
    
    Args:
        historical_data: Historical data with 'ds' and 'y' columns
        forecast_data: Optional forecast data
        
    Returns:
        Plotly figure object
    """
    try:
        # Extract seasonal components
        historical_data['month'] = historical_data['ds'].dt.month
        historical_data['day_of_week'] = historical_data['ds'].dt.dayofweek
        
        # Monthly seasonality
        monthly_avg = historical_data.groupby('month')['y'].mean()
        
        # Weekly seasonality
        weekly_avg = historical_data.groupby('day_of_week')['y'].mean()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Seasonality', 'Weekly Seasonality'),
            vertical_spacing=0.1
        )
        
        # Monthly seasonality
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig.add_trace(
            go.Bar(
                x=[month_names[i-1] for i in monthly_avg.index],
                y=monthly_avg.values,
                name='Monthly Average',
                marker_color='#3498db',
                hovertemplate='<b>%{x}</b><br>Average Sales: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Weekly seasonality
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig.add_trace(
            go.Bar(
                x=[day_names[i] for i in weekly_avg.index],
                y=weekly_avg.values,
                name='Weekly Average',
                marker_color='#e74c3c',
                hovertemplate='<b>%{x}</b><br>Average Sales: $%{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Seasonality Analysis',
                font=dict(size=18, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            height=600,
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_xaxes(title_text="Day of Week", row=2, col=1)
        fig.update_yaxes(title_text="Average Sales ($)", tickformat='$,.0f', row=1, col=1)
        fig.update_yaxes(title_text="Average Sales ($)", tickformat='$,.0f', row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating seasonality chart: {str(e)}")
        return go.Figure()

def calculate_forecast_metrics(historical_data: pd.DataFrame, 
                              forecast_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate business metrics and insights from forecast
    
    Args:
        historical_data: Historical data
        forecast_data: Forecast data
        
    Returns:
        Dictionary with metrics and insights
    """
    try:
        metrics = {}
        
        # Basic forecast statistics
        forecast_mean = forecast_data['forecast'].mean()
        forecast_std = forecast_data['forecast'].std()
        forecast_total = forecast_data['forecast'].sum()
        
        # Historical comparison
        historical_mean = historical_data['y'].mean()
        historical_total = historical_data['y'].sum()
        
        # Growth metrics
        growth_rate = ((forecast_mean - historical_mean) / historical_mean) * 100
        
        # Confidence metrics
        confidence_width = (forecast_data['upper_bound'] - forecast_data['lower_bound']).mean()
        confidence_ratio = confidence_width / forecast_mean * 100
        
        # Trend analysis
        forecast_trend = np.polyfit(range(len(forecast_data)), forecast_data['forecast'], 1)[0]
        
        metrics = {
            'forecast_summary': {
                'total_forecast': forecast_total,
                'average_daily': forecast_mean,
                'std_deviation': forecast_std,
                'growth_rate_percent': growth_rate
            },
            'confidence_metrics': {
                'avg_confidence_width': confidence_width,
                'confidence_ratio_percent': confidence_ratio
            },
            'trend_analysis': {
                'daily_trend': forecast_trend,
                'trend_direction': 'increasing' if forecast_trend > 0 else 'decreasing'
            },
            'business_insights': {
                'forecast_vs_historical': forecast_total / historical_total if historical_total > 0 else 0,
                'volatility_level': 'high' if confidence_ratio > 20 else 'medium' if confidence_ratio > 10 else 'low'
            }
        }
        
        logger.info("Forecast metrics calculated successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating forecast metrics: {str(e)}")
        return {}