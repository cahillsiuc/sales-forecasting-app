# Data Processing Module
# Comprehensive data handling functions for sales forecasting Streamlit app

import pandas as pd
import numpy as np
import io
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, date
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Comprehensive data processing for sales forecasting applications"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.column_mapping = {}
    
    def load_and_validate_file(self, uploaded_file: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and validate uploaded file (CSV or Excel)
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (DataFrame, status_dict)
        """
        status = {
            'success': False,
            'message': '',
            'file_type': '',
            'columns': [],
            'shape': (0, 0)
        }
        
        try:
            # Determine file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            status['file_type'] = file_extension
            
            # Read file based on extension
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Basic validation
            if df.empty:
                raise ValueError("File is empty")
            
            if len(df.columns) < 2:
                raise ValueError("File must have at least 2 columns")
            
            # Update status
            status.update({
                'success': True,
                'message': f"Successfully loaded {file_extension.upper()} file",
                'columns': list(df.columns),
                'shape': df.shape
            })
            
            self.data = df
            logger.info(f"File loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
        except Exception as e:
            status['message'] = f"Error loading file: {str(e)}"
            logger.error(f"File loading error: {str(e)}")
            df = pd.DataFrame()
        
        return df, status
    
    def auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect date, sales, product, and customer columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column types to column names
        """
        column_mapping = {
            'date': None,
            'sales': None,
            'product': None,
            'customer': None
        }
        
        try:
            # Detect date column
            date_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'day', 'month', 'year']):
                    date_candidates.append(col)
                elif df[col].dtype == 'datetime64[ns]':
                    date_candidates.append(col)
            
            if date_candidates:
                # Try to convert to datetime and pick the most successful one
                best_date_col = None
                for col in date_candidates:
                    try:
                        pd.to_datetime(df[col].dropna().iloc[:10])  # Test first 10 non-null values
                        best_date_col = col
                        break
                    except:
                        continue
                column_mapping['date'] = best_date_col
            
            # Detect sales column
            sales_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['sales', 'revenue', 'amount', 'value', 'price']):
                    sales_candidates.append(col)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    sales_candidates.append(col)
            
            if sales_candidates:
                # Pick the first numeric sales-like column
                for col in sales_candidates:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        column_mapping['sales'] = col
                        break
            
            # Detect product column
            product_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['product', 'item', 'sku', 'category', 'type']):
                    product_candidates.append(col)
                elif df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.8:  # Categorical-like
                    product_candidates.append(col)
            
            if product_candidates:
                column_mapping['product'] = product_candidates[0]
            
            # Detect customer column
            customer_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['customer', 'client', 'buyer', 'account']):
                    customer_candidates.append(col)
                elif df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.5:  # High cardinality
                    customer_candidates.append(col)
            
            if customer_candidates:
                column_mapping['customer'] = customer_candidates[0]
            
            self.column_mapping = column_mapping
            logger.info(f"Column mapping detected: {column_mapping}")
            
        except Exception as e:
            logger.error(f"Error in auto column detection: {str(e)}")
        
        return column_mapping
    
    def validate_data_structure(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate data types and formats
        
        Args:
            df: Input DataFrame
            column_mapping: Dictionary mapping column types to names
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'data_types': {},
            'missing_values': {},
            'date_range': None,
            'sales_stats': {}
        }
        
        try:
            # Validate date column
            if column_mapping.get('date'):
                date_col = column_mapping['date']
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    null_dates = df[date_col].isnull().sum()
                    if null_dates > 0:
                        validation_results['warnings'].append(f"{null_dates} invalid dates found in {date_col}")
                    
                    validation_results['date_range'] = {
                        'start': df[date_col].min(),
                        'end': df[date_col].max(),
                        'total_days': (df[date_col].max() - df[date_col].min()).days
                    }
                    validation_results['data_types'][date_col] = 'datetime64[ns]'
                except Exception as e:
                    validation_results['issues'].append(f"Date column {date_col} validation failed: {str(e)}")
                    validation_results['valid'] = False
            
            # Validate sales column
            if column_mapping.get('sales'):
                sales_col = column_mapping['sales']
                try:
                    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
                    null_sales = df[sales_col].isnull().sum()
                    zero_sales = (df[sales_col] == 0).sum()
                    negative_sales = (df[sales_col] < 0).sum()
                    
                    validation_results['sales_stats'] = {
                        'total': df[sales_col].sum(),
                        'mean': df[sales_col].mean(),
                        'median': df[sales_col].median(),
                        'min': df[sales_col].min(),
                        'max': df[sales_col].max(),
                        'null_count': null_sales,
                        'zero_count': zero_sales,
                        'negative_count': negative_sales
                    }
                    
                    if null_sales > 0:
                        validation_results['warnings'].append(f"{null_sales} missing sales values")
                    if zero_sales > len(df) * 0.5:
                        validation_results['warnings'].append(f"High number of zero sales values ({zero_sales})")
                    if negative_sales > 0:
                        validation_results['warnings'].append(f"{negative_sales} negative sales values found")
                    
                    validation_results['data_types'][sales_col] = 'float64'
                except Exception as e:
                    validation_results['issues'].append(f"Sales column {sales_col} validation failed: {str(e)}")
                    validation_results['valid'] = False
            
            # Check for missing values in all columns
            validation_results['missing_values'] = df.isnull().sum().to_dict()
            
            # Overall validation
            if validation_results['issues']:
                validation_results['valid'] = False
            
            logger.info(f"Data validation completed. Valid: {validation_results['valid']}")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            logger.error(f"Data validation error: {str(e)}")
        
        return validation_results
    
    def prepare_forecast_data(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare data for Prophet forecasting (ds, y columns)
        
        Args:
            df: Input DataFrame
            filters: Dictionary of filters to apply
            
        Returns:
            DataFrame formatted for Prophet
        """
        try:
            # Apply filters first
            filtered_df = self.apply_filters(df, filters)
            
            # Get column mapping
            date_col = self.column_mapping.get('date')
            sales_col = self.column_mapping.get('sales')
            
            if not date_col or not sales_col:
                raise ValueError("Date and sales columns must be identified")
            
            # Create Prophet-formatted DataFrame
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(filtered_df[date_col]),
                'y': pd.to_numeric(filtered_df[sales_col], errors='coerce')
            })
            
            # Remove rows with missing values
            prophet_df = prophet_df.dropna()
            
            # Sort by date
            prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
            
            # Remove duplicates (keep last occurrence)
            prophet_df = prophet_df.drop_duplicates(subset=['ds'], keep='last')
            
            # Validate Prophet requirements
            if len(prophet_df) < 2:
                raise ValueError("Insufficient data for forecasting (need at least 2 data points)")
            
            logger.info(f"Prophet data prepared: {len(prophet_df)} rows")
            return prophet_df
            
        except Exception as e:
            logger.error(f"Error preparing Prophet data: {str(e)}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary statistics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': None,
            'sales_summary': {},
            'categorical_summary': {}
        }
        
        try:
            # Date range summary
            date_col = self.column_mapping.get('date')
            if date_col and date_col in df.columns:
                date_series = pd.to_datetime(df[date_col], errors='coerce')
                summary['date_range'] = {
                    'start': date_series.min(),
                    'end': date_series.max(),
                    'span_days': (date_series.max() - date_series.min()).days,
                    'unique_dates': date_series.nunique()
                }
            
            # Sales summary
            sales_col = self.column_mapping.get('sales')
            if sales_col and sales_col in df.columns:
                sales_series = pd.to_numeric(df[sales_col], errors='coerce')
                summary['sales_summary'] = {
                    'total': sales_series.sum(),
                    'mean': sales_series.mean(),
                    'median': sales_series.median(),
                    'std': sales_series.std(),
                    'min': sales_series.min(),
                    'max': sales_series.max(),
                    'zero_count': (sales_series == 0).sum(),
                    'negative_count': (sales_series < 0).sum()
                }
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                summary['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'most_common_count': df[col].value_counts().iloc[0] if not df[col].empty else 0
                }
            
            logger.info(f"Data summary generated for {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
        
        return summary
    
    def apply_filters(self, df: pd.DataFrame, product_filter: Optional[List[str]] = None, 
                     customer_filter: Optional[List[str]] = None, 
                     trend_filter: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply UI filters to data
        
        Args:
            df: Input DataFrame
            product_filter: List of product names to include
            customer_filter: List of customer names to include
            trend_filter: Dictionary with date range filters
            
        Returns:
            Filtered DataFrame
        """
        try:
            filtered_df = df.copy()
            
            # Apply product filter
            if product_filter and self.column_mapping.get('product'):
                product_col = self.column_mapping['product']
                if product_col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[product_col].isin(product_filter)]
            
            # Apply customer filter
            if customer_filter and self.column_mapping.get('customer'):
                customer_col = self.column_mapping['customer']
                if customer_col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[customer_col].isin(customer_filter)]
            
            # Apply date range filter
            if trend_filter and self.column_mapping.get('date'):
                date_col = self.column_mapping['date']
                if date_col in filtered_df.columns:
                    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                    
                    if 'start_date' in trend_filter:
                        start_date = pd.to_datetime(trend_filter['start_date'])
                        filtered_df = filtered_df[filtered_df[date_col] >= start_date]
                    
                    if 'end_date' in trend_filter:
                        end_date = pd.to_datetime(trend_filter['end_date'])
                        filtered_df = filtered_df[filtered_df[date_col] <= end_date]
            
            logger.info(f"Filters applied. Rows: {len(df)} -> {len(filtered_df)}")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return df  # Return original data if filtering fails