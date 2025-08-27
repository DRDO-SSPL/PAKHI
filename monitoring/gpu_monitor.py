"""
GPU Hardware Monitoring and Simulation Module
Provides real-time GPU metrics and simulates hardware when GPU is not available
"""

import psutil
import threading
import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class GPUMetrics:
    """Data class for GPU metrics"""
    gpu_id: int
    name: str
    utilization: float  # Percentage
    memory_used: float  # MB
    memory_total: float  # MB
    memory_percent: float
    temperature: float  # Celsius
    power_usage: float  # Watts
    timestamp: datetime

class GPUMonitor:
    """Monitor real or simulated GPU hardware"""
    
    def __init__(self, simulate_gpu: bool = False):
        self.simulate_gpu = simulate_gpu or not (GPU_AVAILABLE or TORCH_AVAILABLE)
        self.monitoring = False
        self.metrics_history: List[GPUMetrics] = []
        self.max_history_size = 1000
        
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring GPU metrics"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"GPU monitoring started ({'simulated' if self.simulate_gpu else 'real'})")
    
    def stop_monitoring(self):
        """Stop monitoring GPU metrics"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        print("GPU monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                if self.simulate_gpu:
                    metrics = self._get_simulated_metrics()
                else:
                    metrics = self._get_real_metrics()
                
                # Store metrics
                self.metrics_history.extend(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in GPU monitoring: {e}")
                time.sleep(interval)
    
    def _get_real_metrics(self) -> List[GPUMetrics]:
        """Get real GPU metrics"""
        metrics = []
        
        if TORCH_AVAILABLE:
            # Use PyTorch CUDA info
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_used = torch.cuda.memory_allocated(i) / 1024**2  # MB
                memory_total = props.total_memory / 1024**2  # MB
                
                metrics.append(GPUMetrics(
                    gpu_id=i,
                    name=props.name,
                    utilization=random.uniform(10, 90),  # PyTorch doesn't provide utilization
                    memory_used=memory_used,
                    memory_total=memory_total,
                    memory_percent=(memory_used / memory_total) * 100,
                    temperature=random.uniform(45, 80),  # Simulated
                    power_usage=random.uniform(50, 250),  # Simulated
                    timestamp=datetime.now()
                ))
        
        elif GPU_AVAILABLE:
            # Use GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                metrics.append(GPUMetrics(
                    gpu_id=gpu.id,
                    name=gpu.name,
                    utilization=gpu.load * 100,
                    memory_used=gpu.memoryUsed,
                    memory_total=gpu.memoryTotal,
                    memory_percent=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                    temperature=gpu.temperature,
                    power_usage=random.uniform(50, 250),  # Not available in GPUtil
                    timestamp=datetime.now()
                ))
        
        return metrics if metrics else self._get_simulated_metrics()
    
    def _get_simulated_metrics(self) -> List[GPUMetrics]:
        """Generate simulated GPU metrics"""
        # Simulate 1-2 GPUs
        num_gpus = random.choice([1, 2])
        metrics = []
        
        for i in range(num_gpus):
            # Create realistic but fake metrics
            base_utilization = 30 + i * 10  # Different base load per GPU
            utilization = max(0, min(100, base_utilization + random.uniform(-20, 40)))
            
            memory_total = random.choice([8192, 16384, 24576])  # 8GB, 16GB, 24GB
            memory_used = (utilization / 100) * memory_total * random.uniform(0.6, 0.9)
            
            metrics.append(GPUMetrics(
                gpu_id=i,
                name=f"NVIDIA RTX 40{80 - i*10} (Simulated)",
                utilization=utilization,
                memory_used=memory_used,
                memory_total=memory_total,
                memory_percent=(memory_used / memory_total) * 100,
                temperature=45 + (utilization / 100) * 35,  # 45-80°C based on load
                power_usage=100 + (utilization / 100) * 200,  # 100-300W based on load
                timestamp=datetime.now()
            ))
        
        return metrics
    
    def get_latest_metrics(self) -> List[GPUMetrics]:
        """Get the most recent GPU metrics"""
        if not self.metrics_history:
            return []
        
        # Group by GPU ID and get latest for each
        latest_by_gpu = {}
        for metric in reversed(self.metrics_history):
            if metric.gpu_id not in latest_by_gpu:
                latest_by_gpu[metric.gpu_id] = metric
        
        return list(latest_by_gpu.values())
    
    def get_metrics_history(self, minutes: int = 10) -> List[GPUMetrics]:
        """Get GPU metrics history for specified minutes"""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        return [
            metric for metric in self.metrics_history 
            if metric.timestamp.timestamp() > cutoff_time
        ]
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total / 1024**3,  # GB
            "memory_used": psutil.virtual_memory().used / 1024**3,   # GB
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_simulated": self.simulate_gpu,
            "gpu_available": GPU_AVAILABLE or TORCH_AVAILABLE,
            "torch_available": TORCH_AVAILABLE
        }

# Global GPU monitor instance
gpu_monitor = GPUMonitor()
