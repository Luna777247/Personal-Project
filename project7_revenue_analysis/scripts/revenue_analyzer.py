import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RevenueProductAnalyzer:
    """Advanced Revenue and Product Performance Analysis Tool"""

    def __init__(self):
        self.sales_data = None
        self.product_data = None
        self.processed_data = None
        self.kpi_metrics = {}
        self.optimization_recommendations = []

    def generate_mock_sales_data(self, num_records=15000):
        """Generate comprehensive mock sales data with 15,000+ records"""
        print(f"Generating {num_records} mock sales records...")

        # Product categories and items
        categories = {
            'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smart Watch', 'Camera', 'Printer'],
            'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes', 'Hat', 'Socks'],
            'Home & Garden': ['Sofa', 'Table', 'Chair', 'Lamp', 'Bed', 'Refrigerator', 'Washing Machine'],
            'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Magazine', 'Comic', 'Biography'],
            'Sports': ['Basketball', 'Tennis Racket', 'Yoga Mat', 'Dumbbells', 'Bike', 'Swimming Goggles'],
            'Beauty': ['Shampoo', 'Cream', 'Perfume', 'Makeup', 'Nail Polish', 'Hair Dryer'],
            'Food': ['Coffee', 'Tea', 'Chocolate', 'Snacks', 'Beverages', 'Cereals']
        }

        # Flatten products list
        all_products = []
        product_categories = {}
        for category, products in categories.items():
            for product in products:
                all_products.append(f"{category}_{product}")
                product_categories[f"{category}_{product}"] = category

        # Generate data
        np.random.seed(42)
        data = []

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 12, 31)

        for i in range(num_records):
            # Random date within range
            random_days = np.random.randint(0, (end_date - start_date).days)
            order_date = start_date + timedelta(days=random_days)

            # Select random product
            product_name = np.random.choice(all_products)
            category = product_categories[product_name]

            # Generate pricing based on category
            base_prices = {
                'Electronics': (100, 2000),
                'Clothing': (20, 200),
                'Home & Garden': (50, 1500),
                'Books': (10, 50),
                'Sports': (15, 300),
                'Beauty': (5, 100),
                'Food': (2, 20)
            }

            min_price, max_price = base_prices[category]
            unit_price = np.random.uniform(min_price, max_price)

            # Cost calculation (typically 40-70% of selling price)
            cost_percentage = np.random.uniform(0.4, 0.7)
            unit_cost = unit_price * cost_percentage

            # Quantity (1-10 items per order)
            quantity = np.random.randint(1, 11)

            # Customer demographics
            customer_age = np.random.randint(18, 75)
            customer_region = np.random.choice(['North', 'South', 'Central', 'Online'])

            # Order details
            order_id = f"ORD_{20230000 + i}"
            customer_id = f"CUST_{np.random.randint(1000, 9999)}"

            # Calculate totals
            total_revenue = unit_price * quantity
            total_cost = unit_cost * quantity
            profit = total_revenue - total_cost

            # Add some realistic variations
            # Seasonal effects
            month = order_date.month
            if month in [11, 12]:  # Holiday season
                total_revenue *= np.random.uniform(1.1, 1.3)
            elif month in [6, 7, 8]:  # Summer
                if category in ['Sports', 'Food']:
                    total_revenue *= np.random.uniform(1.05, 1.2)

            # Recalculate profit after seasonal adjustments
            profit = total_revenue - total_cost

            data.append({
                'order_id': order_id,
                'customer_id': customer_id,
                'order_date': order_date,
                'product_name': product_name,
                'category': category,
                'unit_price': round(unit_price, 2),
                'unit_cost': round(unit_cost, 2),
                'quantity': quantity,
                'total_revenue': round(total_revenue, 2),
                'total_cost': round(total_cost, 2),
                'profit': round(profit, 2),
                'customer_age': customer_age,
                'customer_region': customer_region,
                'year': order_date.year,
                'month': order_date.month,
                'quarter': (order_date.month - 1) // 3 + 1
            })

        self.sales_data = pd.DataFrame(data)
        print(f"Generated {len(self.sales_data)} sales records")
        return self.sales_data

    def clean_and_process_data(self):
        """Clean and process the sales data"""
        print("Cleaning and processing data...")

        if self.sales_data is None:
            raise ValueError("No sales data available. Generate data first.")

        # Remove duplicates
        initial_count = len(self.sales_data)
        self.sales_data = self.sales_data.drop_duplicates(subset=['order_id'])
        print(f"Removed {initial_count - len(self.sales_data)} duplicate orders")

        # Handle missing values
        self.sales_data = self.sales_data.dropna()

        # Ensure data types
        self.sales_data['order_date'] = pd.to_datetime(self.sales_data['order_date'])
        self.sales_data['year'] = self.sales_data['order_date'].dt.year
        self.sales_data['month'] = self.sales_data['order_date'].dt.month
        self.sales_data['quarter'] = self.sales_data['order_date'].dt.quarter

        # Calculate additional metrics
        self.sales_data['profit_margin'] = (self.sales_data['profit'] / self.sales_data['total_revenue'] * 100).round(2)
        self.sales_data['cost_percentage'] = (self.sales_data['total_cost'] / self.sales_data['total_revenue'] * 100).round(2)

        self.processed_data = self.sales_data.copy()
        print(f"Data processing complete. Final dataset: {len(self.processed_data)} records")
        return self.processed_data

    def calculate_kpi_metrics(self):
        """Calculate key performance indicators"""
        print("Calculating KPI metrics...")

        df = self.processed_data

        # Overall metrics
        total_revenue = df['total_revenue'].sum()
        total_cost = df['total_cost'].sum()
        total_profit = df['profit'].sum()
        overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

        # Monthly revenue
        monthly_revenue = df.groupby(['year', 'month'])['total_revenue'].sum().reset_index()
        monthly_revenue['period'] = monthly_revenue['year'].astype(str) + '-' + monthly_revenue['month'].astype(str).str.zfill(2)

        # Product performance
        product_performance = df.groupby('product_name').agg({
            'total_revenue': 'sum',
            'quantity': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()

        product_performance = product_performance.sort_values('total_revenue', ascending=False)

        # Category performance
        category_performance = df.groupby('category').agg({
            'total_revenue': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).reset_index()

        category_performance['profit_margin'] = (category_performance['profit'] / category_performance['total_revenue'] * 100).round(2)
        category_performance = category_performance.sort_values('total_revenue', ascending=False)

        # Regional performance
        regional_performance = df.groupby('customer_region').agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'order_id': 'count'
        }).reset_index()

        regional_performance['avg_order_value'] = (regional_performance['total_revenue'] / regional_performance['order_id']).round(2)
        regional_performance = regional_performance.sort_values('total_revenue', ascending=False)

        # Quarterly trends
        quarterly_performance = df.groupby(['year', 'quarter']).agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'order_id': 'count'
        }).reset_index()

        quarterly_performance['quarter_label'] = 'Q' + quarterly_performance['quarter'].astype(str) + '-' + quarterly_performance['year'].astype(str)

        self.kpi_metrics = {
            'overall': {
                'total_revenue': round(total_revenue, 2),
                'total_cost': round(total_cost, 2),
                'total_profit': round(total_profit, 2),
                'overall_margin': round(overall_margin, 2),
                'total_orders': len(df),
                'avg_order_value': round(total_revenue / len(df), 2)
            },
            'monthly_revenue': monthly_revenue,
            'product_performance': product_performance,
            'category_performance': category_performance,
            'regional_performance': regional_performance,
            'quarterly_performance': quarterly_performance
        }

        print("KPI calculations complete")
        return self.kpi_metrics

    def identify_underperforming_products(self):
        """Identify products that are underperforming"""
        print("Identifying underperforming products...")

        df = self.processed_data
        product_metrics = df.groupby('product_name').agg({
            'total_revenue': 'sum',
            'quantity': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean',
            'order_id': 'count'
        }).reset_index()

        # Calculate performance scores
        scaler = StandardScaler()
        metrics_for_clustering = product_metrics[['total_revenue', 'quantity', 'profit', 'profit_margin', 'order_id']]
        scaled_metrics = scaler.fit_transform(metrics_for_clustering)

        # K-means clustering to identify performance groups
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        product_metrics['performance_cluster'] = kmeans.fit_predict(scaled_metrics)

        # Identify underperformers (lowest cluster)
        cluster_means = product_metrics.groupby('performance_cluster')[['total_revenue', 'profit']].mean()
        underperforming_cluster = cluster_means['total_revenue'].idxmin()

        underperformers = product_metrics[product_metrics['performance_cluster'] == underperforming_cluster]
        underperformers = underperformers.sort_values('total_revenue', ascending=True)

        return underperformers, product_metrics

    def generate_optimization_recommendations(self):
        """Generate 3 optimization solutions"""
        print("Generating optimization recommendations...")

        underperformers, all_products = self.identify_underperforming_products()

        recommendations = []

        # Recommendation 1: Product Rationalization
        total_underperformer_revenue = underperformers['total_revenue'].sum()
        total_revenue = self.kpi_metrics['overall']['total_revenue']
        underperformer_percentage = (total_underperformer_revenue / total_revenue * 100)

        rec1 = {
            'title': 'Product Portfolio Optimization',
            'description': f'Eliminate or redesign {len(underperformers)} underperforming products',
            'impact': f'Potential revenue impact: ${total_underperformer_revenue:,.0f} ({underperformer_percentage:.1f}% of total revenue)',
            'products_to_review': underperformers['product_name'].head(10).tolist(),
            'actions': [
                'Discontinue products with <5% profit margin',
                'Bundle underperformers with top products',
                'Introduce limited-time promotions for clearance',
                'Gather customer feedback for product improvement'
            ]
        }

        # Recommendation 2: Pricing Strategy
        low_margin_products = all_products[all_products['profit_margin'] < 10]
        potential_price_increase = low_margin_products['total_revenue'].sum() * 0.05  # 5% price increase

        rec2 = {
            'title': 'Dynamic Pricing Strategy',
            'description': f'Optimize pricing for {len(low_margin_products)} low-margin products',
            'impact': f'Potential profit increase: ${potential_price_increase:,.0f} (5% price optimization)',
            'products_to_review': low_margin_products['product_name'].head(10).tolist(),
            'actions': [
                'Implement price elasticity analysis',
                'Introduce tiered pricing models',
                'Dynamic pricing based on demand patterns',
                'Competitive price monitoring and adjustment'
            ]
        }

        # Recommendation 3: Category Focus
        top_categories = self.kpi_metrics['category_performance'].head(3)
        bottom_categories = self.kpi_metrics['category_performance'].tail(3)

        rec3 = {
            'title': 'Category Strategy Rebalancing',
            'description': 'Shift focus from underperforming to high-performing categories',
            'impact': f'Redirect resources from {bottom_categories["category"].tolist()} to {top_categories["category"].tolist()}',
            'category_insights': {
                'top_performers': top_categories[['category', 'total_revenue', 'profit_margin']].to_dict('records'),
                'underperformers': bottom_categories[['category', 'total_revenue', 'profit_margin']].to_dict('records')
            },
            'actions': [
                'Increase marketing budget for top categories by 20%',
                'Reduce inventory investment in bottom categories',
                'Cross-sell products between high and low performing categories',
                'Partner with suppliers for better terms on top categories'
            ]
        }

        self.optimization_recommendations = [rec1, rec2, rec3]
        return self.optimization_recommendations

    def create_visualizations(self):
        """Create comprehensive dashboard visualizations"""
        print("Creating dashboard visualizations...")

        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Revenue & Product Performance Dashboard', fontsize=16, fontweight='bold')

        # 1. Monthly Revenue Trend
        monthly_data = self.kpi_metrics['monthly_revenue']
        axes[0, 0].plot(monthly_data['period'], monthly_data['total_revenue'], marker='o', linewidth=2)
        axes[0, 0].set_title('Monthly Revenue Trend', fontweight='bold')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Category Performance
        category_data = self.kpi_metrics['category_performance']
        bars = axes[0, 1].bar(category_data['category'], category_data['total_revenue'])
        axes[0, 1].set_title('Revenue by Category', fontweight='bold')
        axes[0, 1].set_ylabel('Revenue ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'${height:,.0f}', ha='center', va='bottom', fontsize=8)

        # 3. Profit Margin Distribution
        axes[0, 2].hist(self.processed_data['profit_margin'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('Profit Margin Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Profit Margin (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(self.kpi_metrics['overall']['overall_margin'], color='red', linestyle='--',
                          label=f'Overall: {self.kpi_metrics["overall"]["overall_margin"]:.1f}%')
        axes[0, 2].legend()

        # 4. Top 10 Products
        top_products = self.kpi_metrics['product_performance'].head(10)
        bars = axes[1, 0].barh(top_products['product_name'], top_products['total_revenue'])
        axes[1, 0].set_title('Top 10 Products by Revenue', fontweight='bold')
        axes[1, 0].set_xlabel('Revenue ($)')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1, 0].text(width, bar.get_y() + bar.get_height()/2,
                          f'${width:,.0f}', ha='left', va='center', fontsize=8)

        # 5. Regional Performance
        regional_data = self.kpi_metrics['regional_performance']
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        axes[1, 1].pie(regional_data['total_revenue'], labels=regional_data['customer_region'],
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 1].set_title('Revenue by Region', fontweight='bold')

        # 6. Quarterly Performance
        quarterly_data = self.kpi_metrics['quarterly_performance']
        axes[1, 2].plot(quarterly_data['quarter_label'], quarterly_data['total_revenue'],
                       marker='s', linewidth=2, color='orange')
        axes[1, 2].set_title('Quarterly Revenue Trend', fontweight='bold')
        axes[1, 2].set_ylabel('Revenue ($)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../dashboard/revenue_performance_dashboard.png', dpi=300, bbox_inches='tight')
        print("Dashboard saved to: dashboard/revenue_performance_dashboard.png")
        plt.show()

    def generate_comprehensive_report(self):
        """Generate detailed analysis report"""
        print("Generating comprehensive analysis report...")

        report = f"""
# Revenue & Product Performance Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This comprehensive analysis evaluates revenue and product performance across {len(self.processed_data)} sales transactions, covering multiple product categories and customer segments.

### Key Financial Metrics
- **Total Revenue:** ${self.kpi_metrics['overall']['total_revenue']:,.2f}
- **Total Profit:** ${self.kpi_metrics['overall']['total_profit']:,.2f}
- **Overall Profit Margin:** {self.kpi_metrics['overall']['overall_margin']:.2f}%
- **Total Orders:** {self.kpi_metrics['overall']['total_orders']:,}
- **Average Order Value:** ${self.kpi_metrics['overall']['avg_order_value']:.2f}

## Revenue Analysis

### Monthly Revenue Trends
{self.kpi_metrics['monthly_revenue'].to_string(index=False)}

### Quarterly Performance
{self.kpi_metrics['quarterly_performance'][['quarter_label', 'total_revenue', 'profit', 'order_id']].to_string(index=False)}

## Product Performance Analysis

### Top Performing Products
{self.kpi_metrics['product_performance'].head(10).to_string(index=False)}

### Category Performance
{self.kpi_metrics['category_performance'].to_string(index=False)}

### Regional Performance
{self.kpi_metrics['regional_performance'].to_string(index=False)}

## Optimization Recommendations

"""

        for i, rec in enumerate(self.optimization_recommendations, 1):
            report += f"""
### Recommendation {i}: {rec['title']}
**Description:** {rec['description']}
**Expected Impact:** {rec['impact']}

**Key Actions:**
"""
            for action in rec['actions']:
                report += f"- {action}\n"

            if 'products_to_review' in rec:
                report += f"\n**Products to Review:** {', '.join(rec['products_to_review'][:5])}\n"

        report += """

## Data Quality Summary
- **Total Records Processed:** {:,}
- **Data Completeness:** 100%
- **Duplicate Records Removed:** 0
- **Date Range:** {} to {}

---
*Report generated automatically by Revenue & Product Performance Analyzer*
*Contact: Data Engineering & Business Intelligence Portfolio*
""".format(
            len(self.processed_data),
            self.processed_data['order_date'].min().strftime('%Y-%m-%d'),
            self.processed_data['order_date'].max().strftime('%Y-%m-%d')
        )

        with open('../docs/revenue_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("Comprehensive report saved to: docs/revenue_analysis_report.md")
        return report

    def run_complete_analysis(self):
        """Run the complete revenue and product analysis pipeline"""
        print("=" * 60)
        print("REVENUE & PRODUCT PERFORMANCE ANALYSIS TOOL")
        print("=" * 60)

        try:
            # Step 1: Generate data
            print("\n📊 STEP 1: Data Generation")
            self.generate_mock_sales_data(15000)

            # Step 2: Clean and process
            print("\n🧹 STEP 2: Data Cleaning & Processing")
            self.clean_and_process_data()

            # Step 3: Calculate KPIs
            print("\n📈 STEP 3: KPI Calculations")
            self.calculate_kpi_metrics()

            # Step 4: Generate recommendations
            print("\n💡 STEP 4: Optimization Recommendations")
            self.generate_optimization_recommendations()

            # Step 5: Create visualizations
            print("\n📊 STEP 5: Dashboard Creation")
            self.create_visualizations()

            # Step 6: Generate report
            print("\n📋 STEP 6: Report Generation")
            self.generate_comprehensive_report()

            # Save processed data
            self.processed_data.to_csv('../data/processed_sales_data.csv', index=False)
            print("Processed data saved to: ../data/processed_sales_data.csv")

            print("\n" + "=" * 60)
            print("✅ REVENUE ANALYSIS COMPLETE!")
            print("📁 Check 'dashboard/' folder for visualizations")
            print("📄 Check 'docs/' folder for detailed reports")
            print("📊 Check 'data/' folder for processed datasets")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = RevenueProductAnalyzer()
    analyzer.run_complete_analysis()