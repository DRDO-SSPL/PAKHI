"""
Training Metrics Collection and Monitoring System
Tracks loss, accuracy, and other ML training metrics in real-time
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pickle
import os
from collections import defaultdict, deque

@dataclass
class TrainingMetric:
    """Individual training metric data point"""
    epoch: int
    step: int
    metric_name: str
    value: float
    timestamp: datetime
    model_name: str = "default"
    additional_data: Dict = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'epoch': self.epoch,
            'step': self.step,
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'additional_data': self.additional_data or {}
        }

@dataclass
class TrainingSession:
    """Complete training session information"""
    session_id: str
    model_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    hyperparameters: Dict = None
    metrics: List[TrainingMetric] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []

class TrainingMonitor:
    """Monitor and collect ML training metrics"""
    
    def __init__(self, save_dir: str = "./training_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Current active sessions
        self.active_sessions: Dict[str, TrainingSession] = {}
        
        # All metrics storage
        self.all_metrics: List[TrainingMetric] = []
        self.max_metrics_memory = 10000
        
        # Real-time metrics for each session
        self.session_metrics: Dict[str, List[TrainingMetric]] = defaultdict(list)
        
        # Callbacks for real-time updates
        self.metric_callbacks: List[Callable] = []
        
        # Threading for async operations
        self.save_lock = threading.Lock()
        
    def start_session(self, session_id: str, model_name: str, 
                     hyperparameters: Dict = None) -> TrainingSession:
        """Start a new training session"""
        session = TrainingSession(
            session_id=session_id,
            model_name=model_name,
            start_time=datetime.now(),
            hyperparameters=hyperparameters or {}
        )
        self.active_sessions[session_id] = session
        print(f"Training session started: {session_id} ({model_name})")
        return session
    
    def end_session(self, session_id: str, status: str = "completed"):
        """End a training session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            session.status = status
            
            # Save session data
            self._save_session(session)
            
            print(f"Training session ended: {session_id} ({status})")
            del self.active_sessions[session_id]
    
    def log_metric(self, session_id: str, epoch: int, step: int, 
                   metric_name: str, value: float, additional_data: Dict = None):
        """Log a single training metric"""
        if session_id not in self.active_sessions:
            print(f"Warning: Session {session_id} not found, starting new session")
            self.start_session(session_id, "unknown_model")
        
        session = self.active_sessions[session_id]
        
        metric = TrainingMetric(
            epoch=epoch,
            step=step,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            model_name=session.model_name,
            additional_data=additional_data
        )
        
        # Store in various collections
        self.all_metrics.append(metric)
        session.metrics.append(metric)
        self.session_metrics[session_id].append(metric)
        
        # Limit memory usage
        if len(self.all_metrics) > self.max_metrics_memory:
            self.all_metrics = self.all_metrics[-self.max_metrics_memory:]
        
        # Trigger callbacks for real-time updates
        self._trigger_callbacks(metric)
        
        # Auto-save periodically
        if len(session.metrics) % 100 == 0:
            threading.Thread(target=self._save_session, args=(session,), daemon=True).start()
    
    def log_batch_metrics(self, session_id: str, epoch: int, step: int, 
                         metrics_dict: Dict[str, float], additional_data: Dict = None):
        """Log multiple metrics at once"""
        for metric_name, value in metrics_dict.items():
            self.log_metric(session_id, epoch, step, metric_name, value, additional_data)
    
    def get_session_metrics(self, session_id: str, 
                           metric_names: List[str] = None,
                           last_n: int = None) -> List[TrainingMetric]:
        """Get metrics for a specific session"""
        if session_id not in self.session_metrics:
            return []
        
        metrics = self.session_metrics[session_id]
        
        # Filter by metric names if specified
        if metric_names:
            metrics = [m for m in metrics if m.metric_name in metric_names]
        
        # Get last N metrics if specified
        if last_n:
            metrics = metrics[-last_n:]
        
        return metrics
    
    def get_latest_metrics(self, session_id: str) -> Dict[str, TrainingMetric]:
        """Get the latest value for each metric type in a session"""
        metrics = self.session_metrics.get(session_id, [])
        latest = {}
        
        for metric in reversed(metrics):
            if metric.metric_name not in latest:
                latest[metric.metric_name] = metric
        
        return latest
    
    def get_metric_history(self, session_id: str, metric_name: str, 
                          last_n: int = 100) -> List[TrainingMetric]:
        """Get history for a specific metric"""
        metrics = self.session_metrics.get(session_id, [])
        filtered = [m for m in metrics if m.metric_name == metric_name]
        return filtered[-last_n:]
    
    def get_active_sessions(self) -> Dict[str, TrainingSession]:
        """Get all active training sessions"""
        return self.active_sessions.copy()
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary statistics for a session"""
        if session_id not in self.session_metrics:
            return {}
        
        metrics = self.session_metrics[session_id]
        if not metrics:
            return {}
        
        # Group by metric name
        by_metric = defaultdict(list)
        for metric in metrics:
            by_metric[metric.metric_name].append(metric.value)
        
        summary = {
            'total_metrics': len(metrics),
            'metric_types': list(by_metric.keys()),
            'latest_epoch': max(m.epoch for m in metrics),
            'latest_step': max(m.step for m in metrics),
            'duration_minutes': 0
        }
        
        # Calculate duration
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            duration = datetime.now() - session.start_time
            summary['duration_minutes'] = duration.total_seconds() / 60
        
        # Add metric statistics
        for metric_name, values in by_metric.items():
            summary[f'{metric_name}_latest'] = values[-1]
            summary[f'{metric_name}_min'] = min(values)
            summary[f'{metric_name}_max'] = max(values)
            summary[f'{metric_name}_avg'] = sum(values) / len(values)
        
        return summary
    
    def add_callback(self, callback: Callable[[TrainingMetric], None]):
        """Add callback function to be called when new metrics arrive"""
        self.metric_callbacks.append(callback)
    
    def _trigger_callbacks(self, metric: TrainingMetric):
        """Trigger all registered callbacks"""
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                print(f"Error in metric callback: {e}")
    
    def _save_session(self, session: TrainingSession):
        """Save session data to disk"""
        with self.save_lock:
            try:
                # Save as JSON
                session_data = {
                    'session_id': session.session_id,
                    'model_name': session.model_name,
                    'start_time': session.start_time.isoformat(),
                    'end_time': session.end_time.isoformat() if session.end_time else None,
                    'status': session.status,
                    'hyperparameters': session.hyperparameters,
                    'metrics': [metric.to_dict() for metric in session.metrics]
                }
                
                filename = f"{session.session_id}_{session.model_name}_{session.start_time.strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.save_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
            except Exception as e:
                print(f"Error saving session {session.session_id}: {e}")
    
    def load_session(self, filepath: str) -> Optional[TrainingSession]:
        """Load a saved training session"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct session
            session = TrainingSession(
                session_id=data['session_id'],
                model_name=data['model_name'],
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
                status=data['status'],
                hyperparameters=data['hyperparameters']
            )
            
            # Reconstruct metrics
            for metric_data in data['metrics']:
                metric = TrainingMetric(
                    epoch=metric_data['epoch'],
                    step=metric_data['step'],
                    metric_name=metric_data['metric_name'],
                    value=metric_data['value'],
                    timestamp=datetime.fromisoformat(metric_data['timestamp']),
                    model_name=metric_data['model_name'],
                    additional_data=metric_data.get('additional_data')
                )
                session.metrics.append(metric)
            
            return session
            
        except Exception as e:
            print(f"Error loading session from {filepath}: {e}")
            return None

# Enhanced ML model wrapper for automatic metric collection
class MonitoredModel:
    """Wrapper for ML models to automatically collect training metrics"""
    
    def __init__(self, model, session_id: str, model_name: str, monitor: TrainingMonitor):
        self.model = model
        self.session_id = session_id
        self.model_name = model_name
        self.monitor = monitor
        self.current_epoch = 0
        self.current_step = 0
        
        # Start monitoring session
        self.monitor.start_session(session_id, model_name)
    
    def log_epoch_metrics(self, metrics_dict: Dict[str, float]):
        """Log metrics at the end of an epoch"""
        self.monitor.log_batch_metrics(
            self.session_id, self.current_epoch, self.current_step, metrics_dict
        )
        self.current_epoch += 1
    
    def log_step_metrics(self, metrics_dict: Dict[str, float]):
        """Log metrics at each training step"""
        self.monitor.log_batch_metrics(
            self.session_id, self.current_epoch, self.current_step, metrics_dict
        )
        self.current_step += 1
    
    def finish_training(self, status: str = "completed"):
        """End the training session"""
        self.monitor.end_session(self.session_id, status)

# Global training monitor instance
training_monitor = TrainingMonitor()
