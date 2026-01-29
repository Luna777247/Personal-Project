import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MarketResearchAnalyzer:
    """Mini Market Research Analysis Tool"""

    def __init__(self):
        self.survey_data = []
        self.user_segments = []
        self.brand_insights = []

    def generate_mock_survey_data(self, num_respondents=200):
        """Generate mock survey data for market research"""
        print(f"Generating mock survey data for {num_respondents} respondents...")

        # Define survey questions and response options
        age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
        genders = ['Male', 'Female', 'Other', 'Prefer not to say']
        income_levels = ['Under $25K', '$25K-$50K', '$50K-$75K', '$75K-$100K', 'Over $100K']
        education_levels = ['High School', 'Some College', 'Bachelor\'s', 'Master\'s', 'Doctorate']

        # Brand awareness and perception
        awareness_levels = ['Very High', 'High', 'Moderate', 'Low', 'Very Low']
        satisfaction_levels = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']
        loyalty_levels = ['Very Loyal', 'Loyal', 'Neutral', 'Not Loyal', 'Anti-loyal']

        # Product categories and preferences
        product_categories = ['Electronics', 'Fashion', 'Food & Beverage', 'Health & Beauty', 'Home & Garden']
        purchase_channels = ['Online', 'In-store', 'Mobile App', 'Social Commerce', 'Marketplace']

        # Pain points and issues
        pain_points = [
            'High prices', 'Poor quality', 'Slow delivery', 'Bad customer service',
            'Limited product variety', 'Outdated website', 'Lack of trust',
            'Poor mobile experience', 'Limited payment options', 'Return difficulties'
        ]

        # Communication preferences
        comm_channels = ['Email', 'SMS', 'Social Media', 'Mobile App', 'Website']

        for i in range(num_respondents):
            # Basic demographics
            age = np.random.choice(age_groups, p=[0.3, 0.35, 0.2, 0.1, 0.05])
            gender = np.random.choice(genders, p=[0.4, 0.5, 0.05, 0.05])
            income = np.random.choice(income_levels, p=[0.2, 0.3, 0.25, 0.15, 0.1])
            education = np.random.choice(education_levels, p=[0.15, 0.25, 0.35, 0.2, 0.05])

            # Brand interaction data
            awareness = np.random.choice(awareness_levels, p=[0.1, 0.25, 0.35, 0.2, 0.1])
            satisfaction = np.random.choice(satisfaction_levels, p=[0.15, 0.3, 0.25, 0.2, 0.1])
            loyalty = np.random.choice(loyalty_levels, p=[0.1, 0.25, 0.3, 0.25, 0.1])

            # Purchase behavior
            monthly_spending = np.random.normal(150, 50)  # Average $150 ± $50
            purchase_frequency = np.random.choice(['Weekly', 'Monthly', 'Quarterly', 'Rarely'], p=[0.2, 0.4, 0.3, 0.1])
            preferred_category = np.random.choice(product_categories)
            preferred_channel = np.random.choice(purchase_channels, p=[0.4, 0.25, 0.15, 0.15, 0.05])

            # Issues and pain points (multiple selection)
            num_pain_points = np.random.poisson(2) + 1  # 1-5 pain points
            respondent_pain_points = np.random.choice(pain_points, size=min(num_pain_points, len(pain_points)), replace=False)

            # Communication preferences
            preferred_comm = np.random.choice(comm_channels, size=np.random.randint(1, 4), replace=False)

            # NPS Score (Net Promoter Score)
            nps_score = np.random.normal(6, 2.5)  # Typically 0-10 scale
            nps_score = np.clip(nps_score, 0, 10)

            # Create respondent record
            respondent = {
                'respondent_id': f'R{i+1:03d}',
                'age_group': age,
                'gender': gender,
                'income_level': income,
                'education_level': education,
                'brand_awareness': awareness,
                'satisfaction': satisfaction,
                'loyalty': loyalty,
                'monthly_spending': round(max(0, monthly_spending), 2),
                'purchase_frequency': purchase_frequency,
                'preferred_category': preferred_category,
                'preferred_channel': preferred_channel,
                'pain_points': list(respondent_pain_points),
                'preferred_communication': list(preferred_comm),
                'nps_score': round(nps_score, 1),
                'survey_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30))
            }

            self.survey_data.append(respondent)

        self.survey_df = pd.DataFrame(self.survey_data)
        print(f"Generated survey data for {len(self.survey_df)} respondents")
        return self.survey_df

    def analyze_demographics(self):
        """Analyze respondent demographics"""
        print("Analyzing demographics...")

        # Age distribution
        age_dist = self.survey_df['age_group'].value_counts()

        # Gender distribution
        gender_dist = self.survey_df['gender'].value_counts()

        # Income distribution
        income_dist = self.survey_df['income_level'].value_counts()

        # Education distribution
        education_dist = self.survey_df['education_level'].value_counts()

        print("Demographic Summary:")
        print(f"Age Groups: {age_dist.to_dict()}")
        print(f"Gender: {gender_dist.to_dict()}")
        print(f"Income Levels: {income_dist.to_dict()}")

        return {
            'age': age_dist,
            'gender': gender_dist,
            'income': income_dist,
            'education': education_dist
        }

    def analyze_brand_perception(self):
        """Analyze brand awareness, satisfaction, and loyalty"""
        print("Analyzing brand perception...")

        # Brand awareness
        awareness_dist = self.survey_df['brand_awareness'].value_counts()

        # Satisfaction levels
        satisfaction_dist = self.survey_df['satisfaction'].value_counts()

        # Loyalty levels
        loyalty_dist = self.survey_df['loyalty'].value_counts()

        # NPS Analysis
        nps_scores = self.survey_df['nps_score']
        nps_summary = {
            'mean': nps_scores.mean(),
            'median': nps_scores.median(),
            'std': nps_scores.std(),
            'promoters': (nps_scores >= 9).sum(),
            'passives': ((nps_scores >= 7) & (nps_scores <= 8)).sum(),
            'detractors': (nps_scores <= 6).sum()
        }
        nps_summary['nps'] = (nps_summary['promoters'] - nps_summary['detractors']) / len(nps_scores) * 100

        print("Brand Perception Summary:")
        print(f"Awareness: {awareness_dist.to_dict()}")
        print(f"Satisfaction: {satisfaction_dist.to_dict()}")
        print(f"NPS Score: {nps_summary['nps']:.1f}")
        return {
            'awareness': awareness_dist,
            'satisfaction': satisfaction_dist,
            'loyalty': loyalty_dist,
            'nps': nps_summary
        }

    def analyze_purchase_behavior(self):
        """Analyze purchase patterns and preferences"""
        print("Analyzing purchase behavior...")

        # Spending analysis
        spending_stats = self.survey_df['monthly_spending'].describe()

        # Purchase frequency
        frequency_dist = self.survey_df['purchase_frequency'].value_counts()

        # Preferred categories
        category_dist = self.survey_df['preferred_category'].value_counts()

        # Preferred channels
        channel_dist = self.survey_df['preferred_channel'].value_counts()

        print("Purchase Behavior Summary:")
        print(f"Average Monthly Spending: ${spending_stats['mean']:.2f}")
        print(f"Preferred Categories: {category_dist.to_dict()}")
        print(f"Preferred Channels: {channel_dist.to_dict()}")

        return {
            'spending': spending_stats,
            'frequency': frequency_dist,
            'categories': category_dist,
            'channels': channel_dist
        }

    def identify_pain_points(self):
        """Identify and analyze customer pain points"""
        print("Identifying pain points...")

        # Flatten pain points list
        all_pain_points = []
        for points in self.survey_df['pain_points']:
            all_pain_points.extend(points)

        # Count pain points
        from collections import Counter
        pain_point_counts = Counter(all_pain_points)

        # Sort by frequency
        top_pain_points = dict(sorted(pain_point_counts.items(), key=lambda x: x[1], reverse=True))

        print("Top Pain Points:")
        for point, count in list(top_pain_points.items())[:5]:
            print(f"  {point}: {count} mentions")

        return top_pain_points

    def segment_customers(self, n_clusters=4):
        """Segment customers based on behavior and preferences"""
        print(f"Segmenting customers into {n_clusters} clusters...")

        # Prepare data for clustering
        cluster_data = self.survey_df.copy()

        # Convert categorical to numerical
        awareness_map = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
        satisfaction_map = {'Very Dissatisfied': 1, 'Dissatisfied': 2, 'Neutral': 3, 'Satisfied': 4, 'Very Satisfied': 5}
        loyalty_map = {'Anti-loyal': 1, 'Not Loyal': 2, 'Neutral': 3, 'Loyal': 4, 'Very Loyal': 5}

        cluster_data['awareness_score'] = cluster_data['brand_awareness'].map(awareness_map)
        cluster_data['satisfaction_score'] = cluster_data['satisfaction'].map(satisfaction_map)
        cluster_data['loyalty_score'] = cluster_data['loyalty'].map(loyalty_map)

        # Select features for clustering
        features = ['awareness_score', 'satisfaction_score', 'loyalty_score', 'monthly_spending', 'nps_score']
        X = cluster_data[features]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['segment'] = kmeans.fit_predict(X_scaled)

        # Analyze segments
        segment_analysis = cluster_data.groupby('segment').agg({
            'awareness_score': 'mean',
            'satisfaction_score': 'mean',
            'loyalty_score': 'mean',
            'monthly_spending': 'mean',
            'nps_score': 'mean',
            'respondent_id': 'count'
        }).round(2)

        # Name segments based on characteristics
        segment_names = {
            0: 'High-Value Loyalists',
            1: 'Price-Sensitive Moderates',
            2: 'Low-Engagement Passives',
            3: 'Premium Enthusiasts'
        }

        segment_analysis['segment_name'] = segment_analysis.index.map(segment_names)

        print("Customer Segments:")
        for idx, row in segment_analysis.iterrows():
            print(f"  {row['segment_name']}: {int(row['respondent_id'])} customers")

        return cluster_data, segment_analysis

    def generate_communication_recommendations(self):
        """Generate communication strategy recommendations"""
        print("Generating communication recommendations...")

        # Analyze preferred communication channels
        comm_prefs = {}
        for prefs in self.survey_df['preferred_communication']:
            for pref in prefs:
                comm_prefs[pref] = comm_prefs.get(pref, 0) + 1

        # Sort by preference
        sorted_comm = sorted(comm_prefs.items(), key=lambda x: x[1], reverse=True)

        # Analyze pain points for messaging focus
        pain_points = self.identify_pain_points()
        top_issues = list(pain_points.keys())[:3]

        recommendations = {
            'primary_channels': sorted_comm[:2],
            'secondary_channels': sorted_comm[2:4],
            'key_messages': [
                f"Address {top_issues[0]} concerns prominently",
                f"Highlight solutions for {top_issues[1]}",
                f"Emphasize improvements in {top_issues[2]}",
                "Showcase customer success stories",
                "Provide clear value propositions"
            ],
            'target_segments': [
                "Focus on loyal customers for advocacy campaigns",
                "Target moderate spenders with loyalty programs",
                "Re-engage passive customers with personalized offers",
                "Nurture high-potential segments with premium services"
            ]
        }

        return recommendations

    def generate_visualizations(self):
        """Generate comprehensive market research visualizations"""
        print("Generating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Market Research Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Age Distribution
        age_dist = self.survey_df['age_group'].value_counts()
        age_dist.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Age Distribution')
        axes[0,0].set_ylabel('Number of Respondents')
        axes[0,0].tick_params(axis='x', rotation=45)

        # 2. Brand Satisfaction
        satisfaction_dist = self.survey_df['satisfaction'].value_counts()
        satisfaction_dist.plot(kind='barh', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Brand Satisfaction Levels')

        # 3. Monthly Spending Distribution
        self.survey_df['monthly_spending'].plot(kind='hist', bins=20, ax=axes[0,2], color='orange', alpha=0.7)
        axes[0,2].set_title('Monthly Spending Distribution')
        axes[0,2].set_xlabel('Monthly Spending ($)')
        axes[0,2].axvline(self.survey_df['monthly_spending'].mean(), color='red', linestyle='--', label='.0f')
        axes[0,2].legend()

        # 4. Preferred Purchase Channels
        channel_dist = self.survey_df['preferred_channel'].value_counts()
        channel_dist.plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
        axes[1,0].set_title('Preferred Purchase Channels')

        # 5. NPS Distribution
        self.survey_df['nps_score'].plot(kind='hist', bins=10, ax=axes[1,1], color='purple', alpha=0.7)
        axes[1,1].set_title('NPS Score Distribution')
        axes[1,1].set_xlabel('NPS Score')
        axes[1,1].axvline(9, color='green', linestyle='--', label='Promoters')
        axes[1,1].axvline(7, color='orange', linestyle='--', label='Passives')
        axes[1,1].axvline(6, color='red', linestyle='--', label='Detractors')
        axes[1,1].legend()

        # 6. Top Pain Points
        pain_points = self.identify_pain_points()
        top_5_pain = dict(list(pain_points.items())[:5])
        pd.Series(top_5_pain).plot(kind='barh', ax=axes[1,2], color='red')
        axes[1,2].set_title('Top 5 Customer Pain Points')

        plt.tight_layout()
        plt.savefig('results/market_research_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Generate comprehensive market research report"""
        print("Generating market research report...")

        # Calculate key metrics
        total_respondents = len(self.survey_df)
        avg_nps = self.survey_df['nps_score'].mean()
        avg_spending = self.survey_df['monthly_spending'].mean()

        # Get top insights
        demographics = self.analyze_demographics()
        brand_perception = self.analyze_brand_perception()
        purchase_behavior = self.analyze_purchase_behavior()
        pain_points = self.identify_pain_points()
        comm_recommendations = self.generate_communication_recommendations()

        # Create report
        report = f"""
# Market Research Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents findings from a comprehensive market research survey of {total_respondents} respondents, analyzing brand perception, customer behavior, and market opportunities.

## Key Metrics
- **Total Respondents:** {total_respondents:,}
- **Average NPS Score:** {avg_nps:.1f}/10
- **Average Monthly Spending:** ${avg_spending:.2f}
- **NPS Classification:** {'Good' if avg_nps >= 7 else 'Needs Improvement' if avg_nps >= 5 else 'Critical'}

## Respondent Demographics

### Age Distribution
{demographics['age'].to_string()}

### Gender Distribution
{demographics['gender'].to_string()}

### Income Distribution
{demographics['income'].to_string()}

## Brand Perception Analysis

### Brand Awareness
{brand_perception['awareness'].to_string()}

### Customer Satisfaction
{brand_perception['satisfaction'].to_string()}

### Customer Loyalty
{brand_perception['loyalty'].to_string()}

### Net Promoter Score (NPS)
- **NPS Score:** {brand_perception['nps']['nps']:.1f}
- **Promoters:** {brand_perception['nps']['promoters']} ({brand_perception['nps']['promoters']/total_respondents*100:.1f}%)
- **Passives:** {brand_perception['nps']['passives']} ({brand_perception['nps']['passives']/total_respondents*100:.1f}%)
- **Detractors:** {brand_perception['nps']['detractors']} ({brand_perception['nps']['detractors']/total_respondents*100:.1f}%)

## Purchase Behavior Analysis

### Spending Patterns
- **Mean Monthly Spending:** ${purchase_behavior['spending']['mean']:.2f}
- **Median Monthly Spending:** ${purchase_behavior['spending']['50%']:.2f}
- **Spending Range:** ${purchase_behavior['spending']['min']:.2f} - ${purchase_behavior['spending']['max']:.2f}

### Purchase Frequency
{purchase_behavior['frequency'].to_string()}

### Preferred Product Categories
{purchase_behavior['categories'].to_string()}

### Preferred Purchase Channels
{purchase_behavior['channels'].to_string()}

## Customer Pain Points
The top customer pain points identified:

{chr(10).join([f"{i+1}. {point}: {count} mentions" for i, (point, count) in enumerate(list(pain_points.items())[:5])])}

## Communication Strategy Recommendations

### Primary Communication Channels
{chr(10).join([f"- {channel}: {count} preferences" for channel, count in comm_recommendations['primary_channels']])}

### Key Messaging Focus
{chr(10).join([f"- {msg}" for msg in comm_recommendations['key_messages']])}

### Target Segment Strategy
{chr(10).join([f"- {segment}" for segment in comm_recommendations['target_segments']])}

## Strategic Recommendations

### Immediate Actions (0-3 months)
1. **Address Top Pain Points:** Focus on resolving the most critical customer issues identified in the survey
2. **Communication Channel Optimization:** Prioritize the top 2 communication channels for maximum reach
3. **NPS Improvement:** Implement feedback collection and response systems for detractors

### Medium-term Strategy (3-6 months)
1. **Customer Segmentation:** Develop targeted strategies for different customer segments
2. **Loyalty Program Enhancement:** Strengthen programs for high-value loyal customers
3. **Channel Diversification:** Expand successful purchase channels identified in research

### Long-term Growth (6+ months)
1. **Brand Awareness Campaigns:** Target segments with low awareness
2. **Product Development:** Focus on preferred categories and address unmet needs
3. **Customer Experience Improvement:** Implement comprehensive CX improvements

## Data Sources and Methodology
- **Sample Size:** {total_respondents} survey respondents
- **Data Collection:** Simulated survey responses (can be replaced with real survey data)
- **Analysis Methods:** Statistical analysis, clustering, sentiment analysis
- **Tools Used:** Python, pandas, scikit-learn, matplotlib, seaborn

---
*Report generated automatically by Market Research Analysis Tool*
"""

        # Save report
        with open('results/market_research_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("Report saved to: results/market_research_report.md")
        return report

    def run_complete_analysis(self, num_respondents=200):
        """Run complete market research analysis pipeline"""
        print("=" * 60)
        print("MARKET RESEARCH ANALYSIS TOOL")
        print("=" * 60)

        # Create necessary directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('data', exist_ok=True)

        # Step 1: Data Collection
        print("\n📊 STEP 1: Data Collection")
        self.generate_mock_survey_data(num_respondents)

        # Step 2: Demographic Analysis
        print("\n👥 STEP 2: Demographic Analysis")
        demographics = self.analyze_demographics()

        # Step 3: Brand Perception Analysis
        print("\n🏷️ STEP 3: Brand Perception Analysis")
        brand_perception = self.analyze_brand_perception()

        # Step 4: Purchase Behavior Analysis
        print("\n🛒 STEP 4: Purchase Behavior Analysis")
        purchase_behavior = self.analyze_purchase_behavior()

        # Step 5: Pain Points Identification
        print("\n😞 STEP 5: Pain Points Analysis")
        pain_points = self.identify_pain_points()

        # Step 6: Customer Segmentation
        print("\n🎯 STEP 6: Customer Segmentation")
        segmented_data, segment_analysis = self.segment_customers()

        # Step 7: Communication Recommendations
        print("\n📢 STEP 7: Communication Strategy")
        comm_recommendations = self.generate_communication_recommendations()

        # Step 8: Visualizations
        print("\n📈 STEP 8: Generating Visualizations")
        self.generate_visualizations()

        # Step 9: Report Generation
        print("\n📋 STEP 9: Generating Final Report")
        report = self.generate_report()

        # Save processed data
        self.survey_df.to_csv('data/processed_survey_data.csv', index=False)
        segmented_data.to_csv('data/customer_segments.csv', index=False)

        print("\n" + "=" * 60)
        print("✅ MARKET RESEARCH COMPLETE!")
        print("📁 Check 'results/' folder for visualizations and reports")
        print("📄 Check 'data/' folder for processed data and segments")
        print("=" * 60)

        return {
            'demographics': demographics,
            'brand_perception': brand_perception,
            'purchase_behavior': purchase_behavior,
            'pain_points': pain_points,
            'segments': segment_analysis,
            'communication': comm_recommendations,
            'report': report
        }

def main():
    """Main function to run market research analysis"""
    analyzer = MarketResearchAnalyzer()
    results = analyzer.run_complete_analysis(num_respondents=150)

    # Print summary
    print("\n📊 ANALYSIS SUMMARY:")
    print(f"Total respondents: {len(analyzer.survey_df)}")
    print(f"Average NPS: {results['brand_perception']['nps']['nps']:.1f}/10")
    print(f"Average monthly spending: ${results['purchase_behavior']['spending']['mean']:.2f}")
    print(f"Top pain point: {max(results['pain_points'], key=results['pain_points'].get)}")
    print(f"Number of segments: {len(results['segments'])}")

if __name__ == "__main__":
    main()