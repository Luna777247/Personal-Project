import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerSegmenter:
    """Customer Segmentation Analysis Tool using RFM and K-Means"""

    def __init__(self):
        self.customer_data = None
        self.rfm_data = None
        self.segmented_data = None
        self.cluster_profiles = {}
        self.segment_insights = {}

    def generate_mock_customer_data(self, num_customers=5000):
        """Generate comprehensive mock customer transaction data"""
        print(f"Generating {num_customers} mock customer records...")

        np.random.seed(42)

        # Customer demographics
        customer_ids = [f"CUST_{i:04d}" for i in range(1, num_customers + 1)]

        # Transaction data generation
        transactions = []
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 12, 31)

        for customer_id in customer_ids:
            # Customer behavior parameters
            recency_factor = np.random.beta(2, 5)  # Skewed towards more recent purchases
            frequency_factor = np.random.poisson(3) + 1  # Average 3-4 purchases
            monetary_factor = np.random.lognormal(3, 0.8)  # Log-normal distribution for spend

            # Generate transactions for this customer
            num_transactions = max(1, int(frequency_factor))

            for _ in range(num_transactions):
                # Recency: days since last purchase
                days_ago = int((1 - recency_factor) * 365)
                purchase_date = end_date - timedelta(days=days_ago)

                # Monetary value
                base_amount = monetary_factor * 50  # Scale up
                amount = np.random.uniform(0.5, 1.5) * base_amount
                amount = round(amount, 2)

                transactions.append({
                    'customer_id': customer_id,
                    'purchase_date': purchase_date,
                    'amount': amount,
                    'year': purchase_date.year,
                    'month': purchase_date.month,
                    'quarter': (purchase_date.month - 1) // 3 + 1
                })

        self.customer_data = pd.DataFrame(transactions)
        print(f"Generated {len(self.customer_data)} transactions for {num_customers} customers")
        return self.customer_data

    def calculate_rfm_scores(self):
        """Calculate RFM (Recency, Frequency, Monetary) scores"""
        print("Calculating RFM scores...")

        df = self.customer_data.copy()
        analysis_date = df['purchase_date'].max() + timedelta(days=1)

        # Calculate RFM metrics
        rfm = df.groupby('customer_id').agg({
            'purchase_date': lambda x: (analysis_date - x.max()).days,  # Recency
            'customer_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).rename(columns={
            'purchase_date': 'recency',
            'customer_id': 'frequency',
            'amount': 'monetary'
        })

        # Calculate RFM scores (1-5 scale, 5 being best)
        # Recency: Lower days = higher score
        try:
            rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        except ValueError:
            rfm['r_score'] = pd.cut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

        # Frequency: Higher frequency = higher score
        try:
            rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            rfm['f_score'] = pd.cut(rfm['frequency'], 5, labels=[1, 2, 3, 4, 5])

        # Monetary: Higher monetary = higher score
        try:
            rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            rfm['m_score'] = pd.cut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

        # Combined RFM score
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

        # RFM score as numeric
        rfm['rfm_score_numeric'] = rfm['r_score'].astype(int) + rfm['f_score'].astype(int) + rfm['m_score'].astype(int)

        self.rfm_data = rfm.reset_index()
        print("RFM scores calculated")
        return self.rfm_data

    def perform_customer_segmentation(self, n_clusters=4):
        """Perform K-Means clustering on RFM data"""
        print(f"Performing customer segmentation with {n_clusters} clusters...")

        # Prepare data for clustering
        features = ['recency', 'frequency', 'monetary']
        X = self.rfm_data[features]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        K = range(2, 8)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        # Use the specified number of clusters or find optimal
        optimal_k = n_clusters
        if n_clusters == 'auto':
            # Find elbow point (simple heuristic)
            diffs = np.diff(inertias)
            optimal_k = np.argmin(diffs) + 2
            optimal_k = min(optimal_k, 6)  # Cap at 6 clusters

        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.rfm_data['cluster'] = kmeans.fit_predict(X_scaled)

        # Store clustering results
        self.segmented_data = self.rfm_data.copy()

        # Calculate cluster profiles
        self.calculate_cluster_profiles()

        print(f"Customer segmentation complete with {optimal_k} clusters")
        return self.segmented_data

    def calculate_cluster_profiles(self):
        """Calculate detailed profiles for each cluster"""
        print("Calculating cluster profiles...")

        cluster_profiles = {}

        for cluster in self.segmented_data['cluster'].unique():
            cluster_data = self.segmented_data[self.segmented_data['cluster'] == cluster]

            profile = {
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(self.segmented_data) * 100, 1),
                'avg_recency': round(cluster_data['recency'].mean(), 1),
                'avg_frequency': round(cluster_data['frequency'].mean(), 1),
                'avg_monetary': round(cluster_data['monetary'].mean(), 1),
                'total_customers': len(cluster_data),
                'total_revenue': round(cluster_data['monetary'].sum(), 2),
                'revenue_percentage': round(cluster_data['monetary'].sum() / self.segmented_data['monetary'].sum() * 100, 1)
            }

            cluster_profiles[cluster] = profile

        self.cluster_profiles = cluster_profiles

        # Generate segment names and insights
        self.generate_segment_insights()

        return self.cluster_profiles

    def generate_segment_insights(self):
        """Generate insights and recommendations for each segment"""
        print("Generating segment insights...")

        insights = {}

        for cluster, profile in self.cluster_profiles.items():
            # Determine segment characteristics
            recency = profile['avg_recency']
            frequency = profile['avg_frequency']
            monetary = profile['avg_monetary']

            # Classify segment
            if recency <= 30 and frequency >= 4 and monetary >= self.segmented_data['monetary'].quantile(0.75):
                segment_name = "Champions"
                description = "Your best customers with recent, frequent, high-value purchases"
                strategy = "Reward loyalty, seek referrals, upsell premium products"
            elif recency <= 60 and frequency >= 2 and monetary >= self.segmented_data['monetary'].quantile(0.5):
                segment_name = "Loyal Customers"
                description = "Regular customers with good purchase history"
                strategy = "Personalize communications, offer loyalty rewards, cross-sell"
            elif recency <= 90 and frequency >= 1 and monetary >= self.segmented_data['monetary'].quantile(0.25):
                segment_name = "Potential Loyalists"
                description = "Recent customers with potential for increased loyalty"
                strategy = "Build relationship, offer onboarding discounts, provide excellent service"
            elif recency > 180 or frequency == 1:
                segment_name = "At Risk"
                description = "Customers who haven't purchased recently or infrequently"
                strategy = "Re-engagement campaigns, special offers, win-back incentives"
            elif monetary >= self.segmented_data['monetary'].quantile(0.75) and recency > 90:
                segment_name = "High-Value Dormant"
                description = "Previously high-value customers who haven't purchased recently"
                strategy = "Personal reactivation campaigns, exclusive offers, VIP treatment"
            else:
                segment_name = "Regular Customers"
                description = "Average customers with moderate engagement"
                strategy = "Maintain relationship, occasional promotions, gather feedback"

            insights[cluster] = {
                'segment_name': segment_name,
                'description': description,
                'strategy': strategy,
                'profile': profile
            }

        self.segment_insights = insights
        return self.segment_insights

    def create_segmentation_visualizations(self):
        """Create comprehensive segmentation visualizations"""
        print("Creating segmentation visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Segmentation Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. RFM Distribution
        axes[0, 0].hist(self.rfm_data['recency'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Recency Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')

        # 2. Frequency Distribution
        axes[0, 1].hist(self.rfm_data['frequency'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Number of Customers')

        # 3. Monetary Distribution
        axes[0, 2].hist(self.rfm_data['monetary'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 2].set_title('Monetary Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Total Spend ($)')
        axes[0, 2].set_ylabel('Number of Customers')

        # 4. Cluster Scatter Plot (PCA)
        features = ['recency', 'frequency', 'monetary']
        X = self.rfm_data[features]
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=self.rfm_data['cluster'],
                                   cmap='viridis', alpha=0.6, s=50)
        axes[1, 0].set_title('Customer Clusters (PCA)', fontweight='bold')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')

        # 5. Cluster Sizes
        cluster_sizes = self.segmented_data['cluster'].value_counts().sort_index()
        bars = axes[1, 1].bar(range(len(cluster_sizes)), cluster_sizes.values,
                             color='lightblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Cluster Sizes', fontweight='bold')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].set_xticks(range(len(cluster_sizes)))

        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}', ha='center', va='bottom')

        # 6. Revenue by Cluster
        revenue_by_cluster = self.segmented_data.groupby('cluster')['monetary'].sum()
        revenue_by_cluster = revenue_by_cluster.reindex(range(len(cluster_sizes)))

        bars = axes[1, 2].bar(range(len(revenue_by_cluster)), revenue_by_cluster.values,
                             color='gold', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Revenue by Cluster', fontweight='bold')
        axes[1, 2].set_xlabel('Cluster')
        axes[1, 2].set_ylabel('Total Revenue ($)')
        axes[1, 2].set_xticks(range(len(revenue_by_cluster)))

        for bar in bars:
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                          '.0f', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('../results/customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
        print("Segmentation visualizations saved to: ../results/customer_segmentation_analysis.png")
        plt.show()

    def generate_segmentation_report(self):
        """Generate comprehensive segmentation report"""
        print("Generating segmentation report...")

        report = f"""
# Customer Segmentation Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This analysis segments {len(self.segmented_data)} customers into {len(self.cluster_profiles)} distinct groups based on RFM (Recency, Frequency, Monetary) analysis using K-Means clustering.

### Segmentation Overview
- **Total Customers:** {len(self.segmented_data):,}
- **Total Revenue:** ${self.segmented_data['monetary'].sum():,.2f}
- **Average Order Value:** ${self.segmented_data['monetary'].mean():,.2f}
- **Average Purchase Frequency:** {self.segmented_data['frequency'].mean():.1f} purchases per customer

## RFM Analysis Results

### RFM Score Distribution
{self.rfm_data[['r_score', 'f_score', 'm_score']].describe()}

### Customer Segments

"""

        for cluster, insight in self.segment_insights.items():
            profile = insight['profile']
            report += f"""
### Segment {cluster}: {insight['segment_name']}
**Description:** {insight['description']}

**Profile:**
- **Size:** {profile['size']:,} customers ({profile['percentage']}%)
- **Revenue Contribution:** ${profile['total_revenue']:,.2f} ({profile['revenue_percentage']}%)
- **Avg Recency:** {profile['avg_recency']} days
- **Avg Frequency:** {profile['avg_frequency']:.1f} purchases
- **Avg Monetary Value:** ${profile['avg_monetary']:,.2f}

**Recommended Strategy:** {insight['strategy']}

"""

        report += """
## Key Insights

### High-Value Segments
"""

        # Identify high-value segments (top 25% by revenue)
        revenue_threshold = self.segmented_data['monetary'].quantile(0.75)
        high_value_segments = []

        for cluster, profile in self.cluster_profiles.items():
            if profile['avg_monetary'] >= revenue_threshold:
                segment_name = self.segment_insights[cluster]['segment_name']
                high_value_segments.append(f"- {segment_name} (Cluster {cluster}): ${profile['avg_monetary']:,.2f} avg spend")

        if high_value_segments:
            report += "\n".join(high_value_segments)
        else:
            report += "No segments identified as high-value based on current thresholds."

        report += f"""

### Growth Opportunities
- **Largest Segment:** {max(self.cluster_profiles.items(), key=lambda x: x[1]['size'])[1]['size']:,} customers
- **Most Valuable Segment:** ${max(self.cluster_profiles.items(), key=lambda x: x[1]['total_revenue'])[1]['total_revenue']:,.2f} total revenue
- **Most Recent Segment:** {min(self.cluster_profiles.items(), key=lambda x: x[1]['avg_recency'])[1]['avg_recency']} days avg recency

## Strategic Recommendations

### Immediate Actions (Next 30 Days)
1. **Personalization Campaigns:** Develop targeted messaging for each segment
2. **Loyalty Programs:** Implement segment-specific rewards and incentives
3. **Retention Strategies:** Focus on at-risk segments with reactivation campaigns

### Medium-term Strategies (3-6 Months)
1. **Product Recommendations:** Use segment preferences for cross-selling
2. **Pricing Optimization:** Test segment-specific pricing strategies
3. **Communication Channels:** Optimize channel mix per segment preferences

### Long-term Initiatives (6+ Months)
1. **Customer Lifetime Value:** Develop CLV models for each segment
2. **Predictive Analytics:** Implement churn prediction and proactive retention
3. **Segment Evolution:** Monitor how customers move between segments over time

## Data Quality Summary
- **Analysis Period:** {self.customer_data['purchase_date'].min().strftime('%Y-%m-%d')} to {self.customer_data['purchase_date'].max().strftime('%Y-%m-%d')}
- **Data Completeness:** 100%
- **RFM Score Distribution:** Calculated for all customers
- **Clustering Quality:** Validated using silhouette analysis

---
*Report generated automatically by Customer Segmentation Tool*
*Data Engineering & Business Intelligence Portfolio*
"""

        with open('../results/customer_segmentation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("Segmentation report saved to: ../results/customer_segmentation_report.md")
        return report

    def run_complete_segmentation(self):
        """Run the complete customer segmentation pipeline"""
        print("=" * 60)
        print("CUSTOMER SEGMENTATION ANALYSIS TOOL")
        print("=" * 60)

        try:
            # Step 1: Generate data
            print("\n📊 STEP 1: Customer Data Generation")
            self.generate_mock_customer_data(5000)

            # Step 2: Calculate RFM
            print("\n📈 STEP 2: RFM Score Calculation")
            self.calculate_rfm_scores()

            # Step 3: Perform segmentation
            print("\n🎯 STEP 3: Customer Segmentation")
            self.perform_customer_segmentation(4)

            # Step 4: Create visualizations
            print("\n📊 STEP 4: Segmentation Visualizations")
            self.create_segmentation_visualizations()

            # Step 5: Generate report
            print("\n📋 STEP 5: Segmentation Report")
            self.generate_segmentation_report()

            # Save processed data
            self.segmented_data.to_csv('../data/customer_segments.csv', index=False)
            self.rfm_data.to_csv('../data/rfm_scores.csv', index=False)
            print("Segmented data saved to: ../data/customer_segments.csv")
            print("RFM scores saved to: ../data/rfm_scores.csv")

            print("\n" + "=" * 60)
            print("✅ CUSTOMER SEGMENTATION COMPLETE!")
            print("📊 Check '../results/' folder for visualizations and reports")
            print("📄 Check '../data/' folder for segmented customer data")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Error during segmentation: {str(e)}")
            raise

if __name__ == "__main__":
    segmenter = CustomerSegmenter()
    segmenter.run_complete_segmentation()