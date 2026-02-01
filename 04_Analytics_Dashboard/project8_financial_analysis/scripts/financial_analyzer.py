import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialAnalyzer:
    """Basic Financial Data Analysis Tool"""

    def __init__(self):
        self.financial_data = None
        self.processed_data = None
        self.financial_ratios = {}
        self.growth_analysis = {}
        self.projections = {}

    def generate_mock_financial_data(self, years=3):
        """Generate comprehensive mock financial data"""
        print(f"Generating {years} years of mock financial data...")

        data = []
        np.random.seed(42)

        start_year = 2022
        base_revenue = 1000000  # $1M base revenue
        base_cost = 700000     # $700K base cost

        for year in range(start_year, start_year + years):
            for quarter in range(1, 5):
                # Revenue with growth and seasonality
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * quarter / 4)  # Seasonal variation
                growth_factor = 1.15 ** (year - start_year)  # 15% annual growth
                revenue = base_revenue * growth_factor * seasonal_factor
                revenue *= np.random.uniform(0.9, 1.1)  # Random variation

                # Cost of goods sold (50-65% of revenue)
                cogs_percentage = np.random.uniform(0.5, 0.65)
                cogs = revenue * cogs_percentage

                # Operating expenses (15-25% of revenue)
                opex_percentage = np.random.uniform(0.15, 0.25)
                opex = revenue * opex_percentage

                # Total costs
                total_cost = cogs + opex

                # Profit
                gross_profit = revenue - cogs
                operating_profit = revenue - total_cost
                net_profit = operating_profit * np.random.uniform(0.8, 0.95)  # Tax effect

                # Assets and liabilities (simplified)
                total_assets = revenue * np.random.uniform(0.8, 1.2)
                current_assets = total_assets * np.random.uniform(0.6, 0.8)
                fixed_assets = total_assets - current_assets

                total_liabilities = total_assets * np.random.uniform(0.4, 0.6)
                current_liabilities = total_liabilities * np.random.uniform(0.7, 0.9)
                long_term_debt = total_liabilities - current_liabilities

                equity = total_assets - total_liabilities

                data.append({
                    'year': year,
                    'quarter': quarter,
                    'period': f"Q{quarter}-{year}",
                    'revenue': round(revenue, 2),
                    'cogs': round(cogs, 2),
                    'opex': round(opex, 2),
                    'total_cost': round(total_cost, 2),
                    'gross_profit': round(gross_profit, 2),
                    'operating_profit': round(operating_profit, 2),
                    'net_profit': round(net_profit, 2),
                    'total_assets': round(total_assets, 2),
                    'current_assets': round(current_assets, 2),
                    'fixed_assets': round(fixed_assets, 2),
                    'total_liabilities': round(total_liabilities, 2),
                    'current_liabilities': round(current_liabilities, 2),
                    'long_term_debt': round(long_term_debt, 2),
                    'equity': round(equity, 2)
                })

        self.financial_data = pd.DataFrame(data)
        print(f"Generated {len(self.financial_data)} financial records")
        return self.financial_data

    def calculate_financial_ratios(self):
        """Calculate basic financial ratios"""
        print("Calculating financial ratios...")

        df = self.financial_data.copy()

        # Profitability ratios
        df['gross_margin'] = (df['gross_profit'] / df['revenue'] * 100).round(2)
        df['operating_margin'] = (df['operating_profit'] / df['revenue'] * 100).round(2)
        df['net_margin'] = (df['net_profit'] / df['revenue'] * 100).round(2)

        # Efficiency ratios
        df['asset_turnover'] = (df['revenue'] / df['total_assets']).round(2)

        # Leverage ratios
        df['debt_to_equity'] = (df['total_liabilities'] / df['equity']).round(2)
        df['debt_to_assets'] = (df['total_liabilities'] / df['total_assets'] * 100).round(2)

        # Liquidity ratios
        df['current_ratio'] = (df['current_assets'] / df['current_liabilities']).round(2)

        self.processed_data = df

        # Calculate averages
        self.financial_ratios = {
            'profitability': {
                'avg_gross_margin': df['gross_margin'].mean(),
                'avg_operating_margin': df['operating_margin'].mean(),
                'avg_net_margin': df['net_margin'].mean()
            },
            'efficiency': {
                'avg_asset_turnover': df['asset_turnover'].mean()
            },
            'leverage': {
                'avg_debt_to_equity': df['debt_to_equity'].mean(),
                'avg_debt_to_assets': df['debt_to_assets'].mean()
            },
            'liquidity': {
                'avg_current_ratio': df['current_ratio'].mean()
            }
        }

        print("Financial ratios calculated")
        return self.financial_ratios

    def analyze_growth_rates(self):
        """Analyze growth rates and trends"""
        print("Analyzing growth rates...")

        df = self.processed_data.sort_values(['year', 'quarter'])

        # Quarter-over-quarter growth
        df['revenue_qoq_growth'] = df['revenue'].pct_change() * 100
        df['profit_qoq_growth'] = df['net_profit'].pct_change() * 100

        # Year-over-year growth
        df_yoy = df.copy()
        df_yoy['prev_year_revenue'] = df_yoy.groupby('quarter')['revenue'].shift(1)
        df_yoy['prev_year_profit'] = df_yoy.groupby('quarter')['net_profit'].shift(1)

        df_yoy['revenue_yoy_growth'] = ((df_yoy['revenue'] - df_yoy['prev_year_revenue']) / df_yoy['prev_year_revenue'] * 100).round(2)
        df_yoy['profit_yoy_growth'] = ((df_yoy['net_profit'] - df_yoy['prev_year_profit']) / df_yoy['prev_year_profit'] * 100).round(2)

        # Annual growth rates
        annual_data = df.groupby('year').agg({
            'revenue': 'sum',
            'net_profit': 'sum'
        }).reset_index()

        annual_data['revenue_annual_growth'] = annual_data['revenue'].pct_change() * 100
        annual_data['profit_annual_growth'] = annual_data['net_profit'].pct_change() * 100

        # CAGR calculation
        years = len(annual_data)
        if years > 1:
            start_revenue = annual_data['revenue'].iloc[0]
            end_revenue = annual_data['revenue'].iloc[-1]
            revenue_cagr = (((end_revenue / start_revenue) ** (1 / (years - 1))) - 1) * 100

            start_profit = annual_data['net_profit'].iloc[0]
            end_profit = annual_data['net_profit'].iloc[-1]
            profit_cagr = (((end_profit / start_profit) ** (1 / (years - 1))) - 1) * 100
        else:
            revenue_cagr = profit_cagr = 0

        self.growth_analysis = {
            'quarterly': df[['period', 'revenue_qoq_growth', 'profit_qoq_growth']].dropna(),
            'yearly': df_yoy[['year', 'quarter', 'revenue_yoy_growth', 'profit_yoy_growth']].dropna(),
            'annual': annual_data,
            'cagr': {
                'revenue_cagr': round(revenue_cagr, 2),
                'profit_cagr': round(profit_cagr, 2)
            }
        }

        print("Growth analysis complete")
        return self.growth_analysis

    def generate_financial_projections(self, periods=4):
        """Generate simple financial projections"""
        print(f"Generating {periods} period financial projections...")

        df = self.processed_data.sort_values(['year', 'quarter'])

        # Use linear regression for simple projections
        X = np.arange(len(df)).reshape(-1, 1)
        y_revenue = df['revenue'].values
        y_profit = df['net_profit'].values

        revenue_model = LinearRegression()
        profit_model = LinearRegression()

        revenue_model.fit(X, y_revenue)
        profit_model.fit(X, y_profit)

        # Generate projections
        last_period = len(df)
        projections = []

        for i in range(1, periods + 1):
            period_idx = last_period + i - 1

            projected_revenue = revenue_model.predict([[period_idx]])[0]
            projected_profit = profit_model.predict([[period_idx]])[0]

            # Calculate next period
            year = df['year'].max() + ((df['quarter'].max() + i - 1) // 4)
            quarter = ((df['quarter'].max() + i - 1) % 4) + 1

            projections.append({
                'year': int(year),
                'quarter': int(quarter),
                'period': f"Q{quarter}-{year}",
                'projected_revenue': round(max(0, projected_revenue), 2),
                'projected_profit': round(max(0, projected_profit), 2),
                'projected_margin': round((projected_profit / projected_revenue * 100) if projected_revenue > 0 else 0, 2)
            })

        self.projections = {
            'projections': projections,
            'revenue_r2': round(r2_score(y_revenue, revenue_model.predict(X)), 3),
            'profit_r2': round(r2_score(y_profit, profit_model.predict(X)), 3)
        }

        print("Financial projections generated")
        return self.projections

    def create_financial_charts(self):
        """Create financial analysis charts"""
        print("Creating financial charts...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Financial Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Revenue and Profit Trends
        df = self.processed_data
        axes[0, 0].plot(df['period'], df['revenue'], marker='o', label='Revenue', linewidth=2)
        axes[0, 0].plot(df['period'], df['net_profit'], marker='s', label='Net Profit', linewidth=2)
        axes[0, 0].set_title('Revenue & Profit Trends', fontweight='bold')
        axes[0, 0].set_ylabel('Amount ($)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Profit Margins
        axes[0, 1].plot(df['period'], df['gross_margin'], marker='^', label='Gross Margin', linewidth=2)
        axes[0, 1].plot(df['period'], df['operating_margin'], marker='v', label='Operating Margin', linewidth=2)
        axes[0, 1].plot(df['period'], df['net_margin'], marker='d', label='Net Margin', linewidth=2)
        axes[0, 1].set_title('Profit Margins Over Time', fontweight='bold')
        axes[0, 1].set_ylabel('Margin (%)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Growth Rates
        growth_data = self.growth_analysis['quarterly']
        if len(growth_data) > 0:
            axes[1, 0].bar(growth_data['period'], growth_data['revenue_qoq_growth'],
                          alpha=0.7, label='Revenue Growth', color='skyblue')
            axes[1, 0].bar(growth_data['period'], growth_data['profit_qoq_growth'],
                          alpha=0.7, label='Profit Growth', color='lightcoral', width=0.4)
            axes[1, 0].set_title('Quarter-over-Quarter Growth Rates', fontweight='bold')
            axes[1, 0].set_ylabel('Growth Rate (%)')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Financial Ratios
        ratios = ['gross_margin', 'operating_margin', 'net_margin', 'debt_to_equity', 'current_ratio']
        ratio_labels = ['Gross Margin', 'Operating Margin', 'Net Margin', 'Debt-to-Equity', 'Current Ratio']
        avg_ratios = [df[ratio].mean() for ratio in ratios]

        bars = axes[1, 1].bar(ratio_labels, avg_ratios, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Average Financial Ratios', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)

        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          '.1f', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('../data/financial_analysis_charts.png', dpi=300, bbox_inches='tight')
        print("Charts saved to: data/financial_analysis_charts.png")

    def generate_pdf_report(self):
        """Generate comprehensive 3-page PDF report"""
        print("Generating PDF report...")

        doc = SimpleDocTemplate("../reports/financial_analysis_report.pdf", pagesize=A4)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20
        )

        normal_style = styles['Normal']
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])

        story = []

        # Page 1: Executive Summary
        story.append(Paragraph("Financial Analysis Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", normal_style))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Executive Summary", subtitle_style))

        # Financial overview table
        overview_data = [
            ['Metric', 'Value'],
            ['Analysis Period', f"{self.processed_data['year'].min()}-Q1 to {self.processed_data['year'].max()}-Q4"],
            ['Total Revenue', '${:,.0f}'.format(self.processed_data['revenue'].sum())],
            ['Total Net Profit', '${:,.0f}'.format(self.processed_data['net_profit'].sum())],
            ['Average Net Margin', '{:.1f}%'.format(self.financial_ratios['profitability']['avg_net_margin'])],
            ['Revenue CAGR', '{:.1f}%'.format(self.growth_analysis['cagr']['revenue_cagr'])],
            ['Profit CAGR', '{:.1f}%'.format(self.growth_analysis['cagr']['profit_cagr'])]
        ]

        overview_table = Table(overview_data)
        overview_table.setStyle(table_style)
        story.append(overview_table)
        story.append(Spacer(1, 20))

        # Key ratios
        story.append(Paragraph("Key Financial Ratios", subtitle_style))
        ratios_data = [
            ['Ratio Type', 'Average Value', 'Industry Benchmark*'],
            ['Gross Margin', '{:.1f}%'.format(self.financial_ratios['profitability']['avg_gross_margin']), '20-30%'],
            ['Operating Margin', '{:.1f}%'.format(self.financial_ratios['profitability']['avg_operating_margin']), '10-20%'],
            ['Net Margin', '{:.1f}%'.format(self.financial_ratios['profitability']['avg_net_margin']), '5-15%'],
            ['Current Ratio', '{:.2f}'.format(self.financial_ratios['liquidity']['avg_current_ratio']), '1.5-2.0'],
            ['Debt-to-Equity', '{:.2f}'.format(self.financial_ratios['leverage']['avg_debt_to_equity']), '<1.0']
        ]

        ratios_table = Table(ratios_data)
        ratios_table.setStyle(table_style)
        story.append(ratios_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph("*Industry benchmarks are general guidelines and may vary by sector", normal_style))

        # Page 2: Detailed Analysis
        story.append(Paragraph("Detailed Financial Analysis", title_style))

        story.append(Paragraph("Quarterly Performance", subtitle_style))
        quarterly_data = [list(self.processed_data.columns)]
        for _, row in self.processed_data.iterrows():
            quarterly_data.append([
                row['period'],
                '${:,.0f}'.format(row['revenue']),
                '${:,.0f}'.format(row['net_profit']),
                '{:.1f}%'.format(row['net_margin'])
            ])

        quarterly_table = Table(quarterly_data[:6])  # Limit to first 5 quarters for space
        quarterly_table.setStyle(table_style)
        story.append(quarterly_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("Growth Analysis", subtitle_style))
        growth_data = [
            ['Period', 'Revenue YoY Growth', 'Profit YoY Growth'],
        ]

        for _, row in self.growth_analysis['yearly'].head().iterrows():
            period_label = f"Q{int(row['quarter'])}-{int(row['year'])}"
            growth_data.append([
                period_label,
                '{:.1f}%'.format(row['revenue_yoy_growth']) if pd.notna(row['revenue_yoy_growth']) else 'N/A',
                '{:.1f}%'.format(row['profit_yoy_growth']) if pd.notna(row['profit_yoy_growth']) else 'N/A'
            ])

        growth_table = Table(growth_data)
        growth_table.setStyle(table_style)
        story.append(growth_table)

        # Page 3: Projections and Recommendations
        story.append(Paragraph("Financial Projections & Recommendations", title_style))

        story.append(Paragraph("Future Projections (Next 4 Quarters)", subtitle_style))
        projection_data = [
            ['Period', 'Projected Revenue', 'Projected Profit', 'Projected Margin']
        ]

        for proj in self.projections['projections']:
            projection_data.append([
                proj['period'],
                '${:,.0f}'.format(proj['projected_revenue']),
                '${:,.0f}'.format(proj['projected_profit']),
                '{:.1f}%'.format(proj['projected_margin'])
            ])

        projection_table = Table(projection_data)
        projection_table.setStyle(table_style)
        story.append(projection_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("Key Insights & Recommendations", subtitle_style))
        insights = [
            "• Revenue shows consistent growth with seasonal patterns",
            "• Profit margins are healthy and trending upward",
            "• Working capital management appears efficient",
            "• Growth trajectory supports continued expansion",
            "• Monitor debt levels to maintain financial flexibility"
        ]

        for insight in insights:
            story.append(Paragraph(insight, normal_style))
            story.append(Spacer(1, 5))

        story.append(Spacer(1, 20))
        story.append(Paragraph("Report generated by Financial Analysis Tool", normal_style))
        story.append(Paragraph("Data Engineering & Business Intelligence Portfolio", normal_style))

        doc.build(story)
        print("PDF report saved to: reports/financial_analysis_report.pdf")

    def run_complete_analysis(self):
        """Run the complete financial analysis pipeline"""
        print("=" * 60)
        print("BASIC FINANCIAL DATA ANALYSIS TOOL")
        print("=" * 60)

        try:
            # Step 1: Generate data
            print("\n📊 STEP 1: Financial Data Generation")
            self.generate_mock_financial_data(3)

            # Step 2: Calculate ratios
            print("\n📈 STEP 2: Financial Ratios Calculation")
            self.calculate_financial_ratios()

            # Step 3: Growth analysis
            print("\n📉 STEP 3: Growth Rate Analysis")
            self.analyze_growth_rates()

            # Step 4: Generate projections
            print("\n🔮 STEP 4: Financial Projections")
            self.generate_financial_projections(4)

            # Step 5: Create charts
            print("\n📊 STEP 5: Financial Charts Creation")
            self.create_financial_charts()

            # Step 6: Generate PDF report
            print("\n📄 STEP 6: PDF Report Generation")
            self.generate_pdf_report()

            # Save processed data
            self.processed_data.to_csv('../data/financial_data_processed.csv', index=False)
            print("Processed data saved to: ../data/financial_data_processed.csv")

            print("\n" + "=" * 60)
            print("✅ FINANCIAL ANALYSIS COMPLETE!")
            print("📄 Check 'reports/' folder for PDF report")
            print("📊 Check 'data/' folder for charts and processed data")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = FinancialAnalyzer()
    analyzer.run_complete_analysis()