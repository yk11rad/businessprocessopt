# Install required libraries
!pip install pandas numpy scipy seaborn matplotlib --quiet

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from google.colab import files
import random
from scipy.stats import skewnorm
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration dictionary
CONFIG = {
    'OUTPUT_PATH': 'order_fulfillment_metrics.csv',
    'NUM_ORDERS': 50,
    'HEATMAP_PATH': 'bottleneck_analysis.png',
    'MONTE_CARLO_PATH': 'monte_carlo_risk.png'
}

# Cost and sustainability constants
LABOR_COST_PER_MIN = 0.5  # $0.50/min
ERROR_COST = 10.0         # $10/error
CARBON_PER_MINUTE = 0.2   # kg CO2/min

# Current process time distributions (minutes)
CURRENT_PROCESS = {
    'receive_order': {'mean': 2, 'std': 0.5, 'skew': 0},
    'verify_inventory': {'mean': 7.5, 'std': 1.5, 'skew': 2},  # Right-skewed
    'process_payment': {'mean': 3.5, 'std': 1.0, 'skew': -1},  # Left-skewed
    'pick_items': {'mean': 5, 'std': 1.2, 'skew': 1},
    'pack_order': {'mean': 3, 'std': 0.8, 'skew': 0},
    'create_label': {'mean': 3, 'std': 0.7, 'skew': 1},
    'ship_order': {'mean': 2, 'std': 0.5, 'skew': 0}
}

# Optimized process time distributions (minutes)
OPTIMIZED_PROCESS = {
    'receive_order': {'mean': 1, 'std': 0.2, 'skew': 0},
    'verify_inventory': {'mean': 0.5, 'std': 0.1, 'skew': 0},
    'process_payment': {'mean': 0.5, 'std': 0.1, 'skew': 0},
    'pick_items': {'mean': 2, 'std': 0.5, 'skew': 0},
    'pack_order': {'mean': 2, 'std': 0.4, 'skew': 0},
    'create_label': {'mean': 0.5, 'std': 0.1, 'skew': 0},
    'ship_order': {'mean': 1.5, 'std': 0.3, 'skew': 0}
}

# Helper function for skewed times
def get_step_time(mean, std, skew):
    """Generate step time with skewness."""
    return max(0, skewnorm.rvs(a=skew, loc=mean, scale=std))

# Opportunity cost model
def calculate_opp_cost(delay_minutes):
    """Calculate opportunity cost from delays."""
    base_rate = 0.02  # 2% churn per hour
    try:
        return 1 / (1 - base_rate ** (delay_minutes / 60))
    except ZeroDivisionError:
        return 0.0

# Step 1: Simulate Current Process
def simulate_current_process(order_id):
    """Simulate the current order fulfillment process."""
    logger.info(f"Simulating current process for order {order_id}")
    try:
        times = {}
        for step, params in CURRENT_PROCESS.items():
            times[step] = get_step_time(**params)
        
        total_time = sum(times.values())
        error_prob = 0.1
        errors = sum(1 for step in ['verify_inventory', 'process_payment', 'create_label']
                     if random.random() < error_prob)
        
        labor_cost = total_time * LABOR_COST_PER_MIN
        error_cost = errors * ERROR_COST
        opp_cost = calculate_opp_cost(total_time)
        
        data = {
            'order_id': order_id,
            'process': 'Current',
            **{f"time_{k}": v for k, v in times.items()},
            'total_time': total_time,
            'errors': errors,
            'labor_cost': labor_cost,
            'error_cost': error_cost,
            'opp_cost': opp_cost,
            'total_cost': labor_cost + error_cost + opp_cost,
            'carbon_footprint': total_time * CARBON_PER_MINUTE,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        logger.info(f"Current process order {order_id}: {total_time:.2f} min, {errors} errors")
        return data
    except Exception as e:
        logger.error(f"Current process simulation failed for order {order_id}: {e}")
        raise

# Step 2: Simulate Optimized Process
def simulate_optimized_process(order_id):
    """Simulate the optimized order fulfillment process."""
    logger.info(f"Simulating optimized process for order {order_id}")
    try:
        times = {}
        with ThreadPoolExecutor() as executor:
            inv_future = executor.submit(
                lambda: get_step_time(**OPTIMIZED_PROCESS['verify_inventory'])
            )
            pay_future = executor.submit(
                lambda: get_step_time(**OPTIMIZED_PROCESS['process_payment'])
            )
            times['verify_inventory'] = inv_future.result()
            times['process_payment'] = pay_future.result()
        
        for step in [k for k in OPTIMIZED_PROCESS.keys() if k not in ['verify_inventory', 'process_payment']]:
            times[step] = get_step_time(**OPTIMIZED_PROCESS[step])
        
        total_time = max(times['verify_inventory'], times['process_payment']) + \
                     sum(v for k, v in times.items() if k not in ['verify_inventory', 'process_payment'])
        
        error_prob = 0.02
        errors = sum(1 for step in ['verify_inventory', 'process_payment', 'create_label']
                     if random.random() < error_prob)
        
        # Heuristic picking optimization (simplified)
        if times['pick_items'] > OPTIMIZED_PROCESS['pick_items']['mean'] * 1.5:
            times['pick_items'] *= 0.8  # Reduce time by 20% via optimized route
            total_time = max(times['verify_inventory'], times['process_payment']) + \
                         sum(v for k, v in times.items() if k not in ['verify_inventory', 'process_payment'])
        
        labor_cost = total_time * LABOR_COST_PER_MIN
        error_cost = errors * ERROR_COST
        opp_cost = calculate_opp_cost(total_time)
        
        data = {
            'order_id': order_id,
            'process': 'Optimized',
            **{f"time_{k}": v for k, v in times.items()},
            'total_time': total_time,
            'errors': errors,
            'labor_cost': labor_cost,
            'error_cost': error_cost,
            'opp_cost': opp_cost,
            'total_cost': labor_cost + error_cost + opp_cost,
            'carbon_footprint': total_time * CARBON_PER_MINUTE,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        logger.info(f"Optimized process order {order_id}: {total_time:.2f} min, {errors} errors")
        return data
    except Exception as e:
        logger.error(f"Optimized process simulation failed for order {order_id}: {e}")
        raise

# Step 3: Analyze Bottlenecks
def analyze_bottlenecks(df):
    """Identify process bottlenecks using utilization metrics."""
    logger.info("Analyzing bottlenecks")
    try:
        bottleneck_metrics = []
        for process in ['verify_inventory', 'process_payment', 'pick_items']:
            util = df[f'time_{process}'].mean() / df['total_time'].mean()
            bottleneck_metrics.append((process, util))
        
        bottleneck = max(bottleneck_metrics, key=lambda x: x[1])
        logger.info(f"Major Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1%} utilization)")
        
        # Correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.filter(like='time_').corr(), annot=True, cmap='coolwarm')
        plt.title('Step Time Correlations')
        plt.savefig(CONFIG['HEATMAP_PATH'])
        files.download(CONFIG['HEATMAP_PATH'])
        plt.close()
        
        return bottleneck
    except Exception as e:
        logger.error(f"Bottleneck analysis failed: {e}")
        raise

# Step 4: Monte Carlo Risk Analysis
def monte_carlo_analysis(n=1000):
    """Run Monte Carlo simulations for optimized process risk."""
    logger.info(f"Running Monte Carlo analysis with {n} iterations")
    try:
        results = []
        for _ in range(n):
            data = simulate_optimized_process(0)
            results.append(data['total_time'])
        
        var_95 = np.percentile(results, 95)
        logger.info(f"95% Service Level: {var_95:.1f} minutes")
        
        plt.figure(figsize=(8, 6))
        plt.hist(results, bins=50, color='blue', alpha=0.7)
        plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: {var_95:.1f} min')
        plt.title('Monte Carlo: Optimized Process Time Distribution')
        plt.xlabel('Total Time (minutes)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(CONFIG['MONTE_CARLO_PATH'])
        files.download(CONFIG['MONTE_CARLO_PATH'])
        plt.close()
        
        return var_95
    except Exception as e:
        logger.error(f"Monte Carlo analysis failed: {e}")
        raise

# Step 5: Run Simulations
def run_simulations(num_orders=CONFIG['NUM_ORDERS']):
    """Run simulations for current and optimized processes."""
    logger.info(f"Running simulations for {num_orders} orders")
    try:
        results = []
        
        # Simulate current process
        for i in range(1, num_orders + 1):
            results.append(simulate_current_process(i))
        
        # Simulate optimized process
        for i in range(1, num_orders + 1):
            results.append(simulate_optimized_process(i))
        
        df = pd.DataFrame(results)
        if df.empty:
            logger.error("No simulation data generated")
            raise ValueError("Simulation failed to produce data")
        
        # Analyze bottlenecks
        bottleneck = analyze_bottlenecks(df)
        
        # Monte Carlo analysis
        var_95 = monte_carlo_analysis()
        
        # Sustainability metrics
        carbon_reduction = df[df['process'] == 'Current']['carbon_footprint'].mean() - \
                          df[df['process'] == 'Optimized']['carbon_footprint'].mean()
        logger.info(f"Estimated carbon reduction: {carbon_reduction:.1f} kg CO2/order")
        
        # Save to CSV
        output_path = CONFIG['OUTPUT_PATH']
        df.to_csv(output_path, index=False)
        logger.info(f"Simulation results saved to {output_path}")
        files.download(output_path)
        
        # Summary
        summary = df.groupby('process').agg({
            'total_time': ['mean', 'std'],
            'total_cost': ['mean', 'std'],
            'errors': 'sum',
            'carbon_footprint': 'mean'
        }).round(2)
        logger.info(f"Simulation Summary:\n{summary}")
        
        return df
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

# Main Execution
if __name__ == "__main__":
    logger.info("Starting order fulfillment optimization")
    results = run_simulations()
    print("Simulation Results (First 5):")
    print(results.head())

# README
"""
# Business Process Optimization: Order Fulfillment

## Overview
This Python script applies Lean Six Sigma principles to optimize an order fulfillment process, simulating current and optimized workflows for 50 orders to quantify improvements in time, cost, and environmental impact. It identifies inefficiencies, implements automation, and provides advanced analytics like bottleneck detection and risk assessment, outputting results to a CSV file for Power BI reporting. Designed for scalability and clarity, it showcases process engineering and data analysis skills for professional evaluation.

## Features
- **Process Mapping**: Models order fulfillment (receive, verify inventory, process payment, pick, pack, label, ship) with realistic, skewed time distributions.
- **Simulation Realism**: Uses skewed normal distributions (`skewnorm`) for step times to mimic real-world variability.
- **Bottleneck Analysis**: Identifies critical steps via utilization metrics and visualizes correlations with a heatmap.
- **Cost Modeling**: Includes labor ($0.50/min), error ($10/error), and opportunity costs (nonlinear delay impact).
- **Risk Analysis**: Performs Monte Carlo simulations to estimate 95% service level times, visualized as a histogram.
- **Optimization**: Automates inventory checks, parallelizes payment/inventory, optimizes picking routes, reduces errors from 10% to 2%.
- **Sustainability**: Calculates carbon footprint (0.2 kg CO2/min) and reduction benefits.
- **Output**: Generates a Power BI-compatible CSV (`order_fulfillment_metrics.csv`) with 100 rows, downloadable from Google Colab.
- **Visualization**: Produces heatmap and histogram PNGs for bottleneck and risk analysis.
- **Observability**: Includes detailed logging and summary metrics for transparency.

## Prerequisites
- **Environment**: Google Colab (cloud-based Python notebook).
- **Dependencies**: Automatically installed (`pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`).
- **Internet**: Required for library installation and file downloads.

## Usage
1. Copy and execute the script in a Google Colab notebook.
2. The script will:
   - Simulate current and optimized processes for 50 orders.
   - Analyze bottlenecks and perform Monte Carlo risk assessment.
   - Calculate times, costs, errors, and carbon footprints.
   - Save results to `order_fulfillment_metrics.csv`, downloadable automatically.
   - Generate `bottleneck_analysis.png` and `monte_carlo_risk.png`, also downloadable.
3. Import the CSV into Power BI for visualizations (e.g., time savings, cost reduction).
4. Download `process_optimization.log` for execution details or troubleshooting.

## Process Details
- **Current Process**:
  - Steps: Receive order, verify inventory (manual), process payment (sequential), pick items, pack order, create label (manual), ship order.
  - Inefficiencies: Long inventory checks (7.5 min, right-skewed), payment delays (3.5 min, left-skewed), manual labels (3 min), 10% error rate.
- **Optimized Process**:
  - Improvements: Automated inventory API (0.5 min), parallel payment/inventory, optimized picking (2 min), automated labels (0.5 min), 2% error rate.
  - Approach: Lean (waste elimination), Six Sigma (error reduction).
- **Simulation**: Uses `skewnorm` for time distributions, with costs for labor ($0.50/min), errors ($10/error), and opportunity (delay-based).

## Output
The pipeline produces `order_fulfillment_metrics.csv` with columns:
- `order_id`: Unique identifier (1 to 50).
- `process`: Current or Optimized.
- `time_receive_order`, `time_verify_inventory`, etc.: Step times (minutes).
- `total_time`: Total cycle time.
- `errors`: Number of errors.
- `labor_cost`, `error_cost`, `opp_cost`: Cost components.
- `total_cost`: Sum of all costs.
- `carbon_footprint`: Environmental impact (kg CO2).
- `timestamp`: Simulation timestamp.

Additional outputs:
- `bottleneck_analysis.png`: Heatmap of step time correlations.
- `monte_carlo_risk.png`: Histogram of optimized process times with 95% VaR.

The CSV includes 100 rows (50 current + 50 optimized).

## Technical Details
- **Simulation**: Models steps with skewed normal distributions for realism.
- **Optimization**: Parallelizes inventory/payment using ThreadPoolExecutor; heuristically optimizes picking.
- **Analysis**: Computes bottleneck utilization and Monte Carlo 95% service levels.
- **Cost Model**: Integrates labor, error, and nonlinear opportunity costs.
- **Sustainability**: Quantifies carbon reduction from process improvements.
- **Visualization**: Saves analytical PNGs for easy sharing.
- **Logging**: Captures execution details and summaries (times, costs, errors, carbon).
- **Robustness**: Ensures populated CSV with error checking.

## Customization
To adapt for production:
- **Real Data**: Replace distributions with actual process data (e.g., ERP logs).
- **Steps**: Add steps (e.g., quality checks) in `CURRENT_PROCESS`/`OPTIMIZED_PROCESS`.
- **Costs**: Adjust `LABOR_COST_PER_MIN`, `ERROR_COST`, or opportunity cost formula.
- **Output**: Save to Google Drive or databases (e.g., BigQuery).
- **Scale**: Increase `NUM_ORDERS` for larger simulations.

## Limitations
- **Colab Environment**: Outputs (`order_fulfillment_metrics.csv`, PNGs, log) are temporary and require downloading.
- **Simulation-Based**: Assumes distributions; real data would improve accuracy.
- **Simplifications**: Heuristic picking optimization; no external system integration.

## Future Enhancements
- Integrate ERP APIs for real-time data.
- Use advanced routing algorithms (e.g., vehicle routing) for picking.
- Deploy to cloud platforms (e.g., AWS) for continuous monitoring.
- Add machine learning to predict delays from historical patterns.

## Notes
This system demonstrates expertise in Lean Six Sigma, process simulation, cost modeling, and data visualization, tailored for supply chain optimization. Its design emphasizes analytical rigor, scalability, and clarity, making it an excellent showcase for data engineering and process improvement skills.

For further information, please contact me on linkedin at https://www.linkedin.com/in/edward-antwi-8a01a1196/
"""