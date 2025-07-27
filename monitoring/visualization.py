"""
Real-time Charts and Visualization System
Creates interactive plots and charts for ML training metrics
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading
import json

from training_monitor import TrainingMonitor, TrainingMetric
from gpu_monitor import GPUMonitor

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MetricsVisualizer:
    """Create real-time visualizations for training metrics"""
    
    def __init__(self, training_monitor: TrainingMonitor, gpu_monitor: GPUMonitor):
        self.training_monitor = training_monitor
        self.gpu_monitor = gpu_monitor
        self.figures = {}
        self.update_lock = threading.Lock()
        
    def create_training_dashboard(self, session_id: str, 
                                 figsize: Tuple[int, int] = (15, 10)) -> str:
        """Create a comprehensive training dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Training Dashboard - Session: {session_id}', fontsize=16)
        
        # Get session data
        metrics = self.training_monitor.get_session_metrics(session_id)
        latest_metrics = self.training_monitor.get_latest_metrics(session_id)
        session_summary = self.training_monitor.get_session_summary(session_id)
        
        if not metrics:
            # Show placeholder when no data available
            for i in range(2):
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'No data available', 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].set_title('Waiting for data...')
        else:
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame([
                {
                    'epoch': m.epoch,
                    'step': m.step,
                    'metric_name': m.metric_name,
                    'value': m.value,
                    'timestamp': m.timestamp
                }
                for m in metrics
            ])
            
            # Plot 1: Loss over time
            self._plot_metric_over_time(axes[0, 0], df, 'loss', 'Training Loss')
            
            # Plot 2: Accuracy over time
            self._plot_metric_over_time(axes[0, 1], df, 'accuracy', 'Training Accuracy')
            
            # Plot 3: Learning rate over time
            self._plot_metric_over_time(axes[0, 2], df, 'learning_rate', 'Learning Rate')
            
            # Plot 4: Multiple metrics comparison
            self._plot_multiple_metrics(axes[1, 0], df, ['loss', 'val_loss'], 'Loss Comparison')
            
            # Plot 5: Training progress (epochs)
            self._plot_epoch_progress(axes[1, 1], df, session_summary)
            
            # Plot 6: Current session stats
            self._plot_session_stats(axes[1, 2], latest_metrics, session_summary)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_gpu_dashboard(self, figsize: Tuple[int, int] = (12, 8)) -> str:
        """Create GPU monitoring dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('GPU Monitoring Dashboard', fontsize=16)
        
        # Get GPU data
        gpu_metrics = self.gpu_monitor.get_latest_metrics()
        gpu_history = self.gpu_monitor.get_metrics_history(minutes=10)
        system_info = self.gpu_monitor.get_system_info()
        
        if not gpu_metrics:
            for i in range(2):
                for j in range(2):
                    axes[i, j].text(0.5, 0.5, 'No GPU data available', 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
        else:
            # Plot 1: GPU Utilization
            self._plot_gpu_utilization(axes[0, 0], gpu_metrics)
            
            # Plot 2: GPU Memory Usage
            self._plot_gpu_memory(axes[0, 1], gpu_metrics)
            
            # Plot 3: GPU Temperature and Power
            self._plot_gpu_temp_power(axes[1, 0], gpu_metrics)
            
            # Plot 4: System Overview
            self._plot_system_overview(axes[1, 1], system_info, gpu_metrics)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_live_metric_chart(self, session_id: str, metric_name: str, 
                                last_n: int = 100, figsize: Tuple[int, int] = (10, 6)) -> str:
        """Create a live chart for a specific metric"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get metric history
        history = self.training_monitor.get_metric_history(session_id, metric_name, last_n)
        
        if not history:
            ax.text(0.5, 0.5, f'No data for {metric_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric_name.title()} - No Data')
        else:
            # Plot the metric
            steps = [m.step for m in history]
            values = [m.value for m in history]
            epochs = [m.epoch for m in history]
            
            # Main line plot
            ax.plot(steps, values, 'b-', linewidth=2, alpha=0.8, label=metric_name)
            ax.fill_between(steps, values, alpha=0.3)
            
            # Add epoch markers
            unique_epochs = sorted(set(epochs))
            for epoch in unique_epochs[::max(1, len(unique_epochs)//10)]:  # Show max 10 epoch markers
                epoch_steps = [s for s, e in zip(steps, epochs) if e == epoch]
                if epoch_steps:
                    ax.axvline(x=epoch_steps[0], color='red', linestyle='--', alpha=0.5)
                    ax.text(epoch_steps[0], ax.get_ylim()[1] * 0.9, f'E{epoch}', 
                           rotation=90, ha='right', va='top', fontsize=8)
            
            # Formatting
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric_name.title())
            ax.set_title(f'{metric_name.title()} Over Time (Last {len(history)} points)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add latest value annotation
            if values:
                latest_value = values[-1]
                ax.annotate(f'Latest: {latest_value:.4f}', 
                           xy=(steps[-1], latest_value), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_comparison_chart(self, session_ids: List[str], metric_name: str,
                               figsize: Tuple[int, int] = (12, 6)) -> str:
        """Create a comparison chart between multiple training sessions"""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(session_ids)))
        
        for i, session_id in enumerate(session_ids):
            history = self.training_monitor.get_metric_history(session_id, metric_name, 1000)
            
            if history:
                steps = [m.step for m in history]
                values = [m.value for m in history]
                
                ax.plot(steps, values, color=colors[i], linewidth=2, 
                       label=f'Session {session_id}', alpha=0.8)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric_name.title())
        ax.set_title(f'{metric_name.title()} Comparison Across Sessions')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_metric_over_time(self, ax, df: pd.DataFrame, metric_name: str, title: str):
        """Plot a single metric over time"""
        metric_df = df[df['metric_name'] == metric_name]
        
        if metric_df.empty:
            ax.text(0.5, 0.5, f'No {metric_name} data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.plot(metric_df['step'], metric_df['value'], 'b-', linewidth=2, alpha=0.7)
            ax.fill_between(metric_df['step'], metric_df['value'], alpha=0.3)
            
            # Add trend line
            if len(metric_df) > 5:
                z = np.polyfit(metric_df['step'], metric_df['value'], 1)
                p = np.poly1d(z)
                ax.plot(metric_df['step'], p(metric_df['step']), 'r--', alpha=0.8, linewidth=1)
        
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    def _plot_multiple_metrics(self, ax, df: pd.DataFrame, metric_names: List[str], title: str):
        """Plot multiple metrics on the same chart"""
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, metric_name in enumerate(metric_names):
            metric_df = df[df['metric_name'] == metric_name]
            if not metric_df.empty:
                color = colors[i % len(colors)]
                ax.plot(metric_df['step'], metric_df['value'], 
                       color=color, linewidth=2, alpha=0.7, label=metric_name)
        
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_epoch_progress(self, ax, df: pd.DataFrame, session_summary: Dict):
        """Plot training progress by epochs"""
        if df.empty:
            ax.text(0.5, 0.5, 'No epoch data', ha='center', va='center', transform=ax.transAxes)
        else:
            # Group by epoch and get final step count per epoch
            epoch_progress = df.groupby('epoch')['step'].max().reset_index()
            
            ax.bar(epoch_progress['epoch'], epoch_progress['step'], alpha=0.7, color='skyblue')
            ax.set_title('Steps per Epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Final Step Count')
            ax.grid(True, alpha=0.3)
    
    def _plot_session_stats(self, ax, latest_metrics: Dict, session_summary: Dict):
        """Plot current session statistics"""
        if not latest_metrics:
            ax.text(0.5, 0.5, 'No current stats', ha='center', va='center', transform=ax.transAxes)
        else:
            # Create a simple stats display
            stats_text = []
            
            if 'total_metrics' in session_summary:
                stats_text.append(f"Total Metrics: {session_summary['total_metrics']}")
            
            if 'duration_minutes' in session_summary:
                stats_text.append(f"Duration: {session_summary['duration_minutes']:.1f} min")
            
            for metric_name, metric in latest_metrics.items():
                stats_text.append(f"{metric_name}: {metric.value:.4f}")
            
            # Display as text
            ax.text(0.1, 0.9, '\n'.join(stats_text), transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax.set_title('Current Session Stats')
            ax.axis('off')
    
    def _plot_gpu_utilization(self, ax, gpu_metrics):
        """Plot GPU utilization"""
        gpu_names = [gpu.name for gpu in gpu_metrics]
        utilizations = [gpu.utilization for gpu in gpu_metrics]
        
        bars = ax.bar(gpu_names, utilizations, color='green', alpha=0.7)
        ax.set_title('GPU Utilization (%)')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Utilization %')
        
        # Add value labels on bars
        for bar, util in zip(bars, utilizations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{util:.1f}%', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_gpu_memory(self, ax, gpu_metrics):
        """Plot GPU memory usage"""
        gpu_names = [gpu.name for gpu in gpu_metrics]
        memory_used = [gpu.memory_used / 1024 for gpu in gpu_metrics]  # Convert to GB
        memory_total = [gpu.memory_total / 1024 for gpu in gpu_metrics]
        
        x = np.arange(len(gpu_names))
        width = 0.35
        
        ax.bar(x - width/2, memory_used, width, label='Used', color='orange', alpha=0.7)
        ax.bar(x + width/2, memory_total, width, label='Total', color='lightblue', alpha=0.7)
        
        ax.set_title('GPU Memory Usage (GB)')
        ax.set_xlabel('GPU')
        ax.set_ylabel('Memory (GB)')
        ax.set_xticks(x)
        ax.set_xticklabels(gpu_names, rotation=45, ha='right')
        ax.legend()
    
    def _plot_gpu_temp_power(self, ax, gpu_metrics):
        """Plot GPU temperature and power usage"""
        gpu_names = [gpu.name for gpu in gpu_metrics]
        temperatures = [gpu.temperature for gpu in gpu_metrics]
        power_usage = [gpu.power_usage for gpu in gpu_metrics]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(gpu_names, temperatures, 'r-o', label='Temperature (°C)', linewidth=2)
        line2 = ax2.plot(gpu_names, power_usage, 'b-s', label='Power (W)', linewidth=2)
        
        ax.set_xlabel('GPU')
        ax.set_ylabel('Temperature (°C)', color='red')
        ax2.set_ylabel('Power (W)', color='blue')
        ax.set_title('GPU Temperature & Power')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_system_overview(self, ax, system_info: Dict, gpu_metrics):
        """Plot system overview"""
        # Create a summary of system stats
        stats = [
            f"CPU Cores: {system_info.get('cpu_count', 'N/A')}",
            f"CPU Usage: {system_info.get('cpu_usage', 0):.1f}%",
            f"Memory: {system_info.get('memory_used', 0):.1f}GB / {system_info.get('memory_total', 0):.1f}GB",
            f"GPU Count: {len(gpu_metrics)}",
            f"GPU Simulated: {'Yes' if system_info.get('gpu_simulated') else 'No'}",
            f"PyTorch Available: {'Yes' if system_info.get('torch_available') else 'No'}"
        ]
        
        ax.text(0.1, 0.9, '\n'.join(stats), transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.set_title('System Overview')
        ax.axis('off')
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        with self.update_lock:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close(fig)  # Close figure to free memory
            
            graphic = base64.b64encode(image_png)
            return graphic.decode('utf-8')
    
    def create_plotly_dashboard_data(self, session_id: str) -> Dict:
        """Create data structure for Plotly.js dashboard (for frontend)"""
        metrics = self.training_monitor.get_session_metrics(session_id)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Group metrics by type
        metric_groups = {}
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = {
                    'x': [],  # steps
                    'y': [],  # values
                    'epochs': [],
                    'timestamps': []
                }
            
            metric_groups[metric.metric_name]['x'].append(metric.step)
            metric_groups[metric.metric_name]['y'].append(metric.value)
            metric_groups[metric.metric_name]['epochs'].append(metric.epoch)
            metric_groups[metric.metric_name]['timestamps'].append(metric.timestamp.isoformat())
        
        return {
            'session_id': session_id,
            'metrics': metric_groups,
            'summary': self.training_monitor.get_session_summary(session_id),
            'latest_metrics': {
                name: {
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat()
                }
                for name, metric in self.training_monitor.get_latest_metrics(session_id).items()
            }
        }
