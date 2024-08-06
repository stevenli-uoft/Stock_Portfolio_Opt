import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PortfolioPerformanceVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_performance(self, title="ML-enhanced MVO Vs. Baseline MVO Performance"):
        # Group by Risk Level and calculate averages
        grouped = self.data.groupby('Risk Level').mean().reset_index()

        # Create figure and axis objects with a secondary y-axis
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        # Set width of bars
        bar_width = 0.125

        # Set positions of the bars on X axis
        r1 = np.arange(len(grouped))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]

        # Create bars
        ax1.bar(r1, grouped['ML Predicted'], color='navy', width=bar_width, label='Average of ML Predicted')
        ax1.bar(r2, grouped['ML Actual'], color='skyblue', width=bar_width, label='Average of ML Actual')
        ax1.bar(r3, grouped['Baseline Predicted'], color='darkgreen', width=bar_width,
                label='Average of Baseline Predicted')
        ax1.bar(r4, grouped['Baseline Actual'], color='lightgreen', width=bar_width, label='Average of Baseline Actual')

        # Create lines for Sharpe ratios
        ax2.plot(r2, grouped['ML Sharpe'], color='blue', marker='o', label='Average of ML Sharpe')
        ax2.plot(r3, grouped['Baseline Sharpe'], color='green', marker='o', label='Average of Baseline Sharpe')

        # Add labels and title
        ax1.set_xlabel('Target Volatility')
        ax1.set_ylabel('Portfolio Returns')
        ax2.set_ylabel('Sharpe Ratio')
        plt.title(title, pad=40)

        # Set x-axis ticks
        plt.xticks([r + bar_width for r in range(len(grouped))], grouped['Risk Level'])

        # Add horizontal gridlines
        ax1.grid(which='major', axis='y', linestyle='-', alpha=0.7)
        ax1.grid(which='minor', axis='y', linestyle=':', alpha=0.5)
        ax1.minorticks_on()

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Move legend to the top
        fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.96),
                   ncol=3, fancybox=True, shadow=True)

        # Adjust layout and display the plot
        plt.tight_layout()
        # plt.subplots_adjust(top=0.9)  # Make room for the legend at the top
        plt.show()
