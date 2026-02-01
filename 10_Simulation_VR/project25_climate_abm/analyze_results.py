#!/usr/bin/env python3
"""
Climate Change ABM Results Analyzer

This script analyzes simulation results from GAMA models and generates
visualizations and statistical summaries.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ABMAnalyzer:
    def __init__(self, results_dir="experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def load_simulation_data(self, model_name, run_id):
        """Load simulation results from CSV or JSON files"""
        # This would be adapted based on GAMA's export format
        # For now, create sample data structure
        return self._generate_sample_data(model_name)

    def _generate_sample_data(self, model_name):
        """Generate sample data for demonstration"""
        cycles = 100
        data = {'cycle': list(range(cycles))}

        if model_name == 'flood':
            data.update({
                'flooded_households': [np.random.randint(0, 50) for _ in range(cycles)],
                'evacuated_households': [np.random.randint(0, 30) for _ in range(cycles)],
                'water_level': [5 + i * 0.1 + np.random.normal(0, 1) for i in range(cycles)]
            })
        elif model_name == 'drought':
            data.update({
                'affected_farmers': [np.random.randint(0, 25) for _ in range(cycles)],
                'failed_crops': [np.random.randint(0, 15) for _ in range(cycles)]
            })
        elif model_name == 'rain':
            data.update({
                'flooded_areas': [np.random.randint(0, 40) for _ in range(cycles)],
                'overwhelmed_drainage': [np.random.randint(0, 5) for _ in range(cycles)],
                'total_damage': [i * 0.05 + np.random.normal(0, 0.1) for i in range(cycles)]
            })

        return pd.DataFrame(data)

    def analyze_flood_model(self, data):
        """Analyze flood simulation results"""
        print("=== Flood Model Analysis ===")

        max_flooded = data['flooded_households'].max()
        max_evacuated = data['evacuated_households'].max()
        avg_water_level = data['water_level'].mean()

        print(f"Maximum flooded households: {max_flooded}")
        print(f"Maximum evacuated households: {max_evacuated}")
        print(".2f")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(data['cycle'], data['flooded_households'], label='Flooded', color='red')
        ax1.plot(data['cycle'], data['evacuated_households'], label='Evacuated', color='orange')
        ax1.set_xlabel('Simulation Cycle')
        ax1.set_ylabel('Number of Households')
        ax1.set_title('Household Impact Over Time')
        ax1.legend()

        ax2.plot(data['cycle'], data['water_level'], color='blue')
        ax2.set_xlabel('Simulation Cycle')
        ax2.set_ylabel('Water Level')
        ax2.set_title('Water Level Dynamics')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'flood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'max_flooded': max_flooded,
            'max_evacuated': max_evacuated,
            'avg_water_level': avg_water_level
        }

    def analyze_drought_model(self, data):
        """Analyze drought simulation results"""
        print("=== Drought Model Analysis ===")

        max_affected = data['affected_farmers'].max()
        max_failed = data['failed_crops'].max()

        print(f"Maximum affected farmers: {max_affected}")
        print(f"Maximum failed crops: {max_failed}")

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(data['cycle'], data['affected_farmers'], label='Affected Farmers', color='orange')
        plt.plot(data['cycle'], data['failed_crops'], label='Failed Crops', color='red')
        plt.xlabel('Simulation Cycle')
        plt.ylabel('Count')
        plt.title('Drought Impact on Agriculture')
        plt.legend()
        plt.savefig(self.results_dir / 'drought_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'max_affected_farmers': max_affected,
            'max_failed_crops': max_failed
        }

    def analyze_rain_model(self, data):
        """Analyze heavy rain simulation results"""
        print("=== Heavy Rain Model Analysis ===")

        max_flooded = data['flooded_areas'].max()
        max_overwhelmed = data['overwhelmed_drainage'].max()
        total_damage = data['total_damage'].iloc[-1]

        print(f"Maximum flooded areas: {max_flooded}")
        print(f"Maximum overwhelmed drainage systems: {max_overwhelmed}")
        print(".2f")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(data['cycle'], data['flooded_areas'], label='Flooded Areas', color='blue')
        ax1.plot(data['cycle'], data['overwhelmed_drainage'], label='Overwhelmed Drainage', color='red')
        ax1.set_xlabel('Simulation Cycle')
        ax1.set_ylabel('Count')
        ax1.set_title('Urban Flood Impact')
        ax1.legend()

        ax2.plot(data['cycle'], data['total_damage'], color='purple')
        ax2.set_xlabel('Simulation Cycle')
        ax2.set_ylabel('Total Damage')
        ax2.set_title('Accumulated Damage')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'rain_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'max_flooded_areas': max_flooded,
            'max_overwhelmed_drainage': max_overwhelmed,
            'total_damage': total_damage
        }

    def run_analysis(self, model_name, run_id="sample"):
        """Run complete analysis for a model"""
        data = self.load_simulation_data(model_name, run_id)

        if model_name == 'flood':
            return self.analyze_flood_model(data)
        elif model_name == 'drought':
            return self.analyze_drought_model(data)
        elif model_name == 'rain':
            return self.analyze_rain_model(data)
        else:
            raise ValueError(f"Unknown model: {model_name}")

def main():
    analyzer = ABMAnalyzer()

    # Analyze all models
    models = ['flood', 'drought', 'rain']

    results = {}
    for model in models:
        print(f"\n{'='*50}")
        print(f"Analyzing {model.upper()} MODEL")
        print(f"{'='*50}")
        results[model] = analyzer.run_analysis(model)

    # Save summary
    with open(analyzer.results_dir / 'analysis_summary.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model, stats in results.items():
            json_results[model] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in stats.items()}
        json.dump(json_results, f, indent=2)

    print("\nAnalysis complete! Results saved to experiments/results/")

if __name__ == "__main__":
    main()