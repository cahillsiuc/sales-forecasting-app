# AI Chat Module
# OpenAI integration for sales forecasting insights and business guidance

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import openai
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AIChatAssistant:
    """AI-powered chat assistant specialized in sales forecasting for food manufacturing"""
    
    def __init__(self):
        self.client = None
        self.api_key = None
        self.rate_limit_delay = 1.0  # Delay between requests in seconds
        self.last_request_time = 0
        self.max_retries = 3
        self.system_prompt = """You are an AI sales forecasting expert for food manufacturing. Provide business-focused insights about sales forecasts. Be specific, actionable, and explain patterns in terms of seasonality, market trends, and operational planning. Always mention which forecasting model was used and why it's appropriate. Focus on practical business implications and recommendations for inventory management, production planning, and market strategy."""
        
    def setup_openai_client(self) -> Dict[str, Any]:
        """
        Initialize OpenAI client with error handling
        
        Returns:
            Dictionary with setup status and message
        """
        try:
            # Get API key from environment
            self.api_key = os.getenv('OPENAI_API_KEY')
            
            if not self.api_key or self.api_key == 'your_openai_key_here':
                return {
                    'success': False,
                    'message': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.',
                    'client': None
                }
            
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Test the connection with a simple request
            try:
                test_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                
                logger.info("OpenAI client initialized successfully")
                return {
                    'success': True,
                    'message': 'OpenAI client initialized successfully',
                    'client': self.client
                }
                
            except Exception as e:
                logger.error(f"OpenAI API test failed: {str(e)}")
                return {
                    'success': False,
                    'message': f'OpenAI API connection failed: {str(e)}',
                    'client': None
                }
                
        except Exception as e:
            logger.error(f"Error setting up OpenAI client: {str(e)}")
            return {
                'success': False,
                'message': f'Error initializing OpenAI client: {str(e)}',
                'client': None
            }
    
    def _rate_limit_check(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_forecast_insights(self, forecast_data: pd.DataFrame, 
                                 historical_data: pd.DataFrame,
                                 filters: Dict[str, Any],
                                 model_used: str) -> Dict[str, Any]:
        """
        Create comprehensive context for AI chat based on forecast data
        
        Args:
            forecast_data: Forecast results DataFrame
            historical_data: Historical data DataFrame
            filters: Applied filters dictionary
            model_used: Type of forecasting model used
            
        Returns:
            Dictionary with forecast context and insights
        """
        try:
            # Calculate key metrics
            forecast_total = forecast_data['forecast'].sum()
            forecast_avg = forecast_data['forecast'].mean()
            historical_avg = historical_data['y'].mean() if 'y' in historical_data.columns else 0
            
            # Growth analysis
            growth_rate = ((forecast_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
            
            # Trend analysis
            forecast_trend = np.polyfit(range(len(forecast_data)), forecast_data['forecast'], 1)[0]
            trend_direction = "increasing" if forecast_trend > 0 else "decreasing"
            
            # Seasonality analysis
            date_range = {
                'start': forecast_data['date'].min(),
                'end': forecast_data['date'].max(),
                'periods': len(forecast_data)
            }
            
            # Confidence analysis
            avg_confidence_width = (forecast_data['upper_bound'] - forecast_data['lower_bound']).mean()
            confidence_level = "high" if avg_confidence_width < forecast_avg * 0.1 else "medium" if avg_confidence_width < forecast_avg * 0.2 else "low"
            
            # Filter context
            filter_summary = []
            if filters.get('product_filter'):
                filter_summary.append(f"Products: {', '.join(filters['product_filter'])}")
            if filters.get('customer_filter'):
                filter_summary.append(f"Customers: {', '.join(filters['customer_filter'])}")
            if filters.get('start_date'):
                filter_summary.append(f"From: {filters['start_date']}")
            if filters.get('end_date'):
                filter_summary.append(f"To: {filters['end_date']}")
            
            context = {
                'forecast_summary': {
                    'total_forecast': forecast_total,
                    'average_daily': forecast_avg,
                    'growth_rate_percent': growth_rate,
                    'trend_direction': trend_direction,
                    'confidence_level': confidence_level
                },
                'data_characteristics': {
                    'historical_periods': len(historical_data),
                    'forecast_periods': len(forecast_data),
                    'date_range': date_range,
                    'model_used': model_used
                },
                'filters_applied': filter_summary,
                'business_context': {
                    'industry': 'Food Manufacturing',
                    'forecast_horizon': f"{len(forecast_data)} days",
                    'data_quality': 'high' if len(historical_data) > 30 else 'medium' if len(historical_data) > 10 else 'low'
                }
            }
            
            logger.info("Forecast insights generated successfully")
            return context
            
        except Exception as e:
            logger.error(f"Error generating forecast insights: {str(e)}")
            return {}
    
    def chat_with_ai(self, user_question: str, forecast_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send user question to OpenAI with business-focused context
        
        Args:
            user_question: User's question about the forecast
            forecast_context: Context from generate_forecast_insights
            
        Returns:
            Dictionary with AI response and metadata
        """
        if not self.client:
            setup_result = self.setup_openai_client()
            if not setup_result['success']:
                return {
                    'success': False,
                    'response': setup_result['message'],
                    'fallback': True
                }
        
        try:
            # Rate limiting
            self._rate_limit_check()
            
            # Create context-rich prompt
            context_str = self._format_context_for_prompt(forecast_context)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{context_str}\n\nUser Question: {user_question}"}
            ]
            
            # Make API request with retries
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    ai_response = response.choices[0].message.content.strip()
                    
                    logger.info("AI response generated successfully")
                    return {
                        'success': True,
                        'response': ai_response,
                        'model_used': 'gpt-3.5-turbo',
                        'tokens_used': response.usage.total_tokens if response.usage else 0,
                        'fallback': False
                    }
                    
                except openai.RateLimitError:
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"API request failed (attempt {attempt + 1}), retrying...")
                        time.sleep(1)
                        continue
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Error in AI chat: {str(e)}")
            return {
                'success': False,
                'response': self._get_fallback_response(user_question, forecast_context),
                'fallback': True,
                'error': str(e)
            }
    
    def _format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """Format forecast context for AI prompt"""
        try:
            context_parts = []
            
            # Forecast summary
            if 'forecast_summary' in context:
                summary = context['forecast_summary']
                context_parts.append(f"Forecast Summary:")
                context_parts.append(f"- Total Forecast: ${summary.get('total_forecast', 0):,.2f}")
                context_parts.append(f"- Average Daily: ${summary.get('average_daily', 0):,.2f}")
                context_parts.append(f"- Growth Rate: {summary.get('growth_rate_percent', 0):.1f}%")
                context_parts.append(f"- Trend: {summary.get('trend_direction', 'stable')}")
                context_parts.append(f"- Confidence: {summary.get('confidence_level', 'medium')}")
            
            # Model and data info
            if 'data_characteristics' in context:
                data_info = context['data_characteristics']
                context_parts.append(f"\nModel Used: {data_info.get('model_used', 'Unknown')}")
                context_parts.append(f"Historical Data Points: {data_info.get('historical_periods', 0)}")
                context_parts.append(f"Forecast Period: {data_info.get('forecast_periods', 0)} days")
            
            # Filters
            if 'filters_applied' in context and context['filters_applied']:
                context_parts.append(f"\nFilters Applied: {', '.join(context['filters_applied'])}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return "Context formatting error"
    
    def _get_fallback_response(self, question: str, context: Dict[str, Any]) -> str:
        """Provide fallback response when OpenAI API fails"""
        try:
            # Simple keyword-based responses
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['trend', 'direction', 'going']):
                trend = context.get('forecast_summary', {}).get('trend_direction', 'stable')
                return f"Based on the forecast data, the trend appears to be {trend}. This suggests that sales are moving in a {trend} direction over the forecast period."
            
            elif any(word in question_lower for word in ['confidence', 'accurate', 'reliable']):
                confidence = context.get('forecast_summary', {}).get('confidence_level', 'medium')
                return f"The forecast confidence level is {confidence}. This indicates the reliability of the predictions based on historical data patterns."
            
            elif any(word in question_lower for word in ['seasonal', 'seasonality', 'pattern']):
                return "Seasonal patterns in food manufacturing often relate to holidays, weather changes, and consumer behavior. The forecast model accounts for these patterns to provide accurate predictions."
            
            elif any(word in question_lower for word in ['model', 'algorithm', 'method']):
                model = context.get('data_characteristics', {}).get('model_used', 'Unknown')
                return f"The {model} model was used for this forecast. This model is well-suited for time series data with seasonal patterns common in food manufacturing."
            
            else:
                return "I'm currently unable to process your question due to technical limitations. Please try rephrasing your question or contact support for assistance."
                
        except Exception as e:
            logger.error(f"Error in fallback response: {str(e)}")
            return "I'm experiencing technical difficulties. Please try again later."
    
    def get_predefined_questions(self) -> List[Dict[str, str]]:
        """
        Return list of common forecasting questions for food manufacturing
        
        Returns:
            List of question dictionaries with categories
        """
        return [
            {
                'category': 'Trend Analysis',
                'questions': [
                    "What is the overall trend in our sales forecast?",
                    "Are we seeing growth or decline in the forecast period?",
                    "How does this forecast compare to our historical performance?",
                    "What factors might be driving the trend we're seeing?"
                ]
            },
            {
                'category': 'Seasonality & Patterns',
                'questions': [
                    "What seasonal patterns do you see in our forecast?",
                    "How do holidays and seasonal events affect our sales?",
                    "Are there any unusual patterns in the forecast data?",
                    "What time of year typically shows the highest sales?"
                ]
            },
            {
                'category': 'Business Planning',
                'questions': [
                    "What should we focus on for inventory management?",
                    "How should we adjust our production planning?",
                    "What marketing strategies would be most effective?",
                    "How confident should we be in these forecasts?"
                ]
            },
            {
                'category': 'Model & Methodology',
                'questions': [
                    "Why was this forecasting model chosen?",
                    "How accurate is this forecasting method?",
                    "What are the limitations of this forecast?",
                    "Should we consider different forecasting approaches?"
                ]
            },
            {
                'category': 'Market Insights',
                'questions': [
                    "What market trends are reflected in this forecast?",
                    "How might external factors affect our sales?",
                    "What competitive advantages can we leverage?",
                    "How should we respond to market changes?"
                ]
            }
        ]
    
    def explain_model_choice(self, model_type: str, data_characteristics: Dict[str, Any]) -> str:
        """
        AI explanation of why a specific model was chosen
        
        Args:
            model_type: Type of forecasting model used
            data_characteristics: Characteristics of the input data
            
        Returns:
            Explanation of model choice
        """
        try:
            # Get data characteristics
            data_points = data_characteristics.get('historical_periods', 0)
            forecast_periods = data_characteristics.get('forecast_periods', 0)
            data_quality = data_characteristics.get('data_quality', 'medium')
            
            explanations = {
                'prophet': f"""
The Prophet model was chosen for this forecast because:

1. **Seasonality Handling**: Prophet excels at capturing seasonal patterns common in food manufacturing, such as holiday spikes, weather-related demand changes, and consumer behavior cycles.

2. **Data Robustness**: With {data_points} historical data points, Prophet can effectively identify and model complex seasonal and trend patterns.

3. **Business-Friendly**: Prophet provides interpretable components (trend, seasonality) that are easy to explain to business stakeholders.

4. **Food Industry Fit**: Prophet handles the irregular patterns typical in food sales, including promotional effects and external events.
                """,
                
                'ensemble': f"""
The Ensemble model was selected because:

1. **Data Complexity**: With {data_points} data points, the ensemble approach combines ARIMA's statistical rigor with linear trend analysis for robust predictions.

2. **Balanced Approach**: The weighted combination (70% ARIMA, 30% trend) provides both statistical accuracy and business interpretability.

3. **Uncertainty Handling**: Ensemble methods naturally provide better uncertainty quantification, crucial for food manufacturing planning.

4. **Adaptability**: This approach works well when data shows both trend and seasonal components.
                """,
                
                'linear': f"""
The Linear Regression model was chosen because:

1. **Simplicity & Interpretability**: Linear regression provides clear, explainable relationships between time-based features and sales.

2. **Feature Engineering**: The model incorporates time-based features (day of year, month, day of week) that capture seasonal patterns in food sales.

3. **Computational Efficiency**: Fast training and prediction make it suitable for real-time forecasting applications.

4. **Baseline Performance**: Provides a solid baseline that can be compared against more complex models.
                """
            }
            
            base_explanation = explanations.get(model_type.lower(), "The selected model provides appropriate forecasting capabilities for your data.")
            
            # Add data-specific insights
            if data_quality == 'high':
                base_explanation += "\n\nWith high-quality historical data, this model can provide reliable forecasts for business planning."
            elif data_quality == 'low':
                base_explanation += "\n\nNote: With limited historical data, consider collecting more data points for improved forecast accuracy."
            
            return base_explanation.strip()
            
        except Exception as e:
            logger.error(f"Error explaining model choice: {str(e)}")
            return f"The {model_type} model was selected based on the characteristics of your data and forecasting requirements."

# Convenience functions for easy integration
def setup_openai_client() -> Dict[str, Any]:
    """Convenience function to setup OpenAI client"""
    assistant = AIChatAssistant()
    return assistant.setup_openai_client()

def get_predefined_questions() -> List[Dict[str, str]]:
    """Convenience function to get predefined questions"""
    assistant = AIChatAssistant()
    return assistant.get_predefined_questions()

def explain_model_choice(model_type: str, data_characteristics: Dict[str, Any]) -> str:
    """Convenience function to explain model choice"""
    assistant = AIChatAssistant()
    return assistant.explain_model_choice(model_type, data_characteristics)