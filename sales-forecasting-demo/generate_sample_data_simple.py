# Sample Sales Data Generator (Simple Version)
# Creates realistic frozen food sales data using only standard Python libraries

import csv
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class SalesDataGenerator:
    """Generates realistic frozen food sales data"""
    
    def __init__(self):
        self.setup_random_seed()
        self.setup_products()
        self.setup_customers()
        self.setup_seasonal_patterns()
    
    def setup_random_seed(self):
        """Set random seed for reproducible results"""
        np.random.seed(42)
        random.seed(42)
    
    def setup_products(self):
        """Define product catalog with realistic frozen food items"""
        self.products = [
            # Vegetables (Regular trend, steady year-round)
            {"id": "FF0001", "description": "Frozen Peas", "category": "Vegetables", "trend_type": "Regular"},
            {"id": "FF0002", "description": "Frozen Corn", "category": "Vegetables", "trend_type": "Regular"},
            {"id": "FF0003", "description": "Frozen Carrots", "category": "Vegetables", "trend_type": "Regular"},
            {"id": "FF0004", "description": "Frozen Spinach", "category": "Vegetables", "trend_type": "Regular"},
            {"id": "FF0005", "description": "Frozen Broccoli", "category": "Vegetables", "trend_type": "Regular"},
            {"id": "FF0006", "description": "Frozen Cauliflower", "category": "Vegetables", "trend_type": "Regular"},
            
            # Prepared Foods (Regular trend, winter/holiday spikes)
            {"id": "FF0007", "description": "Frozen Pizza", "category": "Prepared Foods", "trend_type": "Regular"},
            {"id": "FF0008", "description": "Frozen Lasagna", "category": "Prepared Foods", "trend_type": "Regular"},
            {"id": "FF0009", "description": "Frozen Meatballs", "category": "Prepared Foods", "trend_type": "Regular"},
            {"id": "FF0010", "description": "Frozen Chicken Nuggets", "category": "Prepared Foods", "trend_type": "Regular"},
            {"id": "FF0011", "description": "Frozen Fish Fillets", "category": "Prepared Foods", "trend_type": "Regular"},
            {"id": "FF0012", "description": "Frozen French Fries", "category": "Prepared Foods", "trend_type": "Regular"},
            
            # Desserts (Regular trend, summer spikes)
            {"id": "FF0013", "description": "Frozen Ice Cream", "category": "Desserts", "trend_type": "Regular"},
            {"id": "FF0014", "description": "Frozen Waffles", "category": "Desserts", "trend_type": "Regular"},
            
            # Irregular items (unpredictable demand)
            {"id": "FF0015", "description": "Frozen Shrimp", "category": "Seafood", "trend_type": "Irregular"},
            {"id": "FF0016", "description": "Frozen Specialty Vegetables", "category": "Vegetables", "trend_type": "Irregular"},
            {"id": "FF0017", "description": "Frozen Organic Mix", "category": "Organic", "trend_type": "Irregular"},
            {"id": "FF0018", "description": "Frozen Gluten-Free Pizza", "category": "Specialty", "trend_type": "Irregular"},
            {"id": "FF0019", "description": "Frozen Vegan Burgers", "category": "Plant-Based", "trend_type": "Irregular"},
            {"id": "FF0020", "description": "Frozen Seasonal Items", "category": "Seasonal", "trend_type": "Irregular"}
        ]
    
    def setup_customers(self):
        """Define realistic customer names"""
        self.customers = [
            "FreshMart Grocery",
            "HealthFoods Market", 
            "SuperValue Stores",
            "Gourmet Foods Co",
            "Family Market Chain",
            "Organic Corner Store",
            "Metro Food Center",
            "Village Grocery",
            "Premium Market",
            "Budget Foods Inc",
            "Natural Foods Store",
            "City Market Place",
            "Farm Fresh Foods",
            "Quick Stop Market",
            "Elite Grocery Co"
        ]
    
    def setup_seasonal_patterns(self):
        """Define seasonal multipliers for different categories"""
        # Monthly multipliers (Jan=0, Feb=1, ..., Dec=11)
        self.seasonal_patterns = {
            "Vegetables": [1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.0, 1.0],  # Steady year-round
            "Prepared Foods": [1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4],  # Winter/holiday spikes
            "Desserts": [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.4, 1.3, 1.1, 1.0, 0.9, 0.8],  # Summer spikes
            "Seafood": [1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1],  # Slight winter preference
            "Organic": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Steady
            "Specialty": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Steady
            "Plant-Based": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Steady
            "Seasonal": [1.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.5]  # Winter/Christmas spikes
        }
    
    def generate_base_sales(self, product: Dict, month: int) -> float:
        """Generate base sales value for a product in a given month"""
        # Base sales by category (monthly averages)
        base_sales = {
            "Vegetables": 1500,
            "Prepared Foods": 2000,
            "Desserts": 1200,
            "Seafood": 800,
            "Organic": 600,
            "Specialty": 400,
            "Plant-Based": 500,
            "Seasonal": 300
        }
        
        category = product["category"]
        base_value = base_sales.get(category, 1000)
        
        # Apply seasonal multiplier
        seasonal_multiplier = self.seasonal_patterns[category][month - 1]
        
        # Add random variation (±10-15% for regular, 0-50% for irregular)
        if product["trend_type"] == "Regular":
            variation = np.random.normal(0, 0.12)  # ±12% standard deviation
        else:  # Irregular
            variation = np.random.uniform(-0.25, 0.25)  # ±25% uniform distribution
        
        sales_value = base_value * seasonal_multiplier * (1 + variation)
        
        # For irregular items, 50% chance of zero sales
        if product["trend_type"] == "Irregular" and random.random() < 0.5:
            sales_value = 0
        
        return max(0, sales_value)  # Ensure non-negative
    
    def generate_customer_allocation(self, total_sales: float, num_customers: int) -> List[float]:
        """Generate realistic customer allocation of sales"""
        # Create customer weights (some customers buy more than others)
        customer_weights = np.random.gamma(2, 1, num_customers)  # Gamma distribution for realistic spread
        customer_weights = customer_weights / customer_weights.sum()  # Normalize to sum to 1
        
        # Allocate sales proportionally
        customer_sales = total_sales * customer_weights
        
        # Add some randomness to make it more realistic
        customer_sales = customer_sales * np.random.normal(1, 0.1, num_customers)
        
        return np.maximum(0, customer_sales)  # Ensure non-negative
    
    def generate_date_range(self) -> List[datetime]:
        """Generate monthly date range from 2024-01-01 to 2025-12-01"""
        dates = []
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 12, 1)
        
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return dates
    
    def generate_sales_data(self) -> List[Dict]:
        """Generate complete sales dataset"""
        print("Generating realistic frozen food sales data...")
        
        data = []
        dates = self.generate_date_range()
        
        for product in self.products:
            print(f"Processing {product['description']}...")
            
            for date in dates:
                month = date.month
                
                # Generate base sales for this product-month
                base_sales = self.generate_base_sales(product, month)
                
                # Determine number of customers for this product-month (10-15 customers)
                num_customers = random.randint(10, 15)
                selected_customers = random.sample(self.customers, num_customers)
                
                # Generate customer allocations
                customer_sales = self.generate_customer_allocation(base_sales, num_customers)
                
                # Create rows for each customer
                for i, customer in enumerate(selected_customers):
                    sales_value = customer_sales[i]
                    
                    # Only include rows with actual sales (skip zero sales)
                    if sales_value > 0:
                        data.append({
                            'ItemID': product['id'],
                            'Description': product['description'],
                            'TrendType': product['trend_type'],
                            'ds': date.strftime('%Y-%m-%d'),
                            'CustomerName': customer,
                            'y': round(sales_value, 2)
                        })
        
        # Sort by date and item
        data.sort(key=lambda x: (x['ds'], x['ItemID'], x['CustomerName']))
        
        print(f"Generated {len(data)} sales records")
        print(f"Date range: {data[0]['ds']} to {data[-1]['ds']}")
        
        return data
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Save data to CSV file"""
        if not data:
            print("No data to save!")
            return
        
        fieldnames = ['ItemID', 'Description', 'TrendType', 'ds', 'CustomerName', 'y']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✅ Data saved to: {filename}")
    
    def generate_summary_stats(self, data: List[Dict]):
        """Generate and display summary statistics"""
        print("\n" + "="*50)
        print("SAMPLE DATA SUMMARY")
        print("="*50)
        
        # Overall stats
        total_sales = sum(row['y'] for row in data)
        print(f"Total Records: {len(data):,}")
        print(f"Total Sales: ${total_sales:,.2f}")
        
        # Count unique items
        unique_items = len(set(row['ItemID'] for row in data))
        unique_customers = len(set(row['CustomerName'] for row in data))
        print(f"Unique Products: {unique_items}")
        print(f"Unique Customers: {unique_customers}")
        
        # By trend type
        regular_sales = sum(row['y'] for row in data if row['TrendType'] == 'Regular')
        irregular_sales = sum(row['y'] for row in data if row['TrendType'] == 'Irregular')
        regular_count = len([row for row in data if row['TrendType'] == 'Regular'])
        irregular_count = len([row for row in data if row['TrendType'] == 'Irregular'])
        
        print(f"\nBy Trend Type:")
        print(f"  Regular: {regular_count:,} records, ${regular_sales:,.2f} sales")
        print(f"  Irregular: {irregular_count:,} records, ${irregular_sales:,.2f} sales")
        
        # Top products by sales
        product_sales = {}
        for row in data:
            desc = row['Description']
            product_sales[desc] = product_sales.get(desc, 0) + row['y']
        
        print(f"\nTop Products by Sales:")
        sorted_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:10]
        for product, sales in sorted_products:
            print(f"  {product}: ${sales:,.2f}")
        
        # Monthly sales pattern
        monthly_sales = {}
        for row in data:
            month = row['ds'][:7]  # YYYY-MM
            monthly_sales[month] = monthly_sales.get(month, 0) + row['y']
        
        print(f"\nMonthly Sales Pattern:")
        for month in sorted(monthly_sales.keys()):
            print(f"  {month}: ${monthly_sales[month]:,.2f}")

def main():
    """Main function to generate and save sample data"""
    generator = SalesDataGenerator()
    
    # Generate the data
    sales_data = generator.generate_sales_data()
    
    # Display summary statistics
    generator.generate_summary_stats(sales_data)
    
    # Save to CSV
    output_file = "sample_data/sample_sales.csv"
    generator.save_to_csv(sales_data, output_file)
    
    print(f"\n📊 Ready for AI Sales Forecasting Demo!")
    
    # Display first few rows
    print(f"\nFirst 10 rows of generated data:")
    for i, row in enumerate(sales_data[:10]):
        print(f"{i+1:2d}. {row['ItemID']} | {row['Description']:<25} | {row['TrendType']:<8} | {row['ds']} | {row['CustomerName']:<20} | ${row['y']:>8.2f}")

if __name__ == "__main__":
    main()