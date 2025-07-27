"""
Example ML Training Script with Integrated Monitoring
Demonstrates how to use the monitoring system with real ML training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import random
import uuid
from datetime import datetime

# Import monitoring components
from training_monitor import training_monitor, MonitoredModel
from gpu_monitor import gpu_monitor

def create_sample_cnn():
    """Create a simple CNN for MNIST"""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return SimpleCNN()

def get_mnist_data(batch_size=64):
    """Get MNIST data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def calculate_accuracy(model, data_loader, device):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    model.train()
    return 100 * correct / total

def train_monitored_model(epochs=5, batch_size=64, learning_rate=0.001):
    """Train a model with full monitoring integration"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create unique session ID
    session_id = f"mnist_cnn_{uuid.uuid4().hex[:8]}"
    model_name = "MNIST_CNN_Demo"
    
    # Initialize model and training components
    model = create_sample_cnn().to(device)
    train_loader, test_loader = get_mnist_data(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    
    # Start monitoring session
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "architecture": "SimpleCNN",
        "dataset": "MNIST"
    }
    
    training_monitor.start_session(session_id, model_name, hyperparameters)
    print(f"Started monitoring session: {session_id}")
    
    try:
        # Training loop
        step = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate batch accuracy
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_accuracy = 100 * batch_correct / labels.size(0)
                
                # Log metrics for this step
                current_lr = scheduler.get_last_lr()[0]
                step_metrics = {
                    "loss": loss.item(),
                    "accuracy": batch_accuracy,
                    "learning_rate": current_lr
                }
                
                training_monitor.log_batch_metrics(
                    session_id, epoch, step, step_metrics
                )
                
                # Accumulate epoch stats
                epoch_loss += loss.item()
                epoch_correct += batch_correct
                epoch_total += labels.size(0)
                
                step += 1
                
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx:4d}/{len(train_loader)}: "
                          f"Loss: {loss.item():.4f}, Acc: {batch_accuracy:.2f}%")
            
            # Calculate epoch metrics
            epoch_loss /= len(train_loader)
            epoch_accuracy = 100 * epoch_correct / epoch_total
            
            # Calculate validation accuracy
            val_accuracy = calculate_accuracy(model, test_loader, device)
            
            # Update learning rate
            scheduler.step()
            
            # Log epoch metrics
            epoch_metrics = {
                "epoch_loss": epoch_loss,
                "epoch_accuracy": epoch_accuracy,
                "val_accuracy": val_accuracy,
                "learning_rate": scheduler.get_last_lr()[0]
            }
            
            training_monitor.log_batch_metrics(
                session_id, epoch, step, epoch_metrics
            )
            
            print(f"  Epoch Summary:")
            print(f"    Train Loss: {epoch_loss:.4f}")
            print(f"    Train Acc:  {epoch_accuracy:.2f}%")
            print(f"    Val Acc:    {val_accuracy:.2f}%")
            print(f"    LR:         {scheduler.get_last_lr()[0]:.6f}")
            
            # Simulate some processing time
            time.sleep(1)
        
        # Calculate final test accuracy
        final_accuracy = calculate_accuracy(model, test_loader, device)
        print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")
        
        # Log final metrics
        training_monitor.log_metric(
            session_id, epochs-1, step, "final_test_accuracy", final_accuracy
        )
        
        # End monitoring session successfully
        training_monitor.end_session(session_id, "completed")
        print(f"Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        training_monitor.end_session(session_id, "failed")
        raise
    
    return session_id, model

def simulate_multiple_training_runs():
    """Simulate multiple training runs for demonstration"""
    print("Starting multiple training simulations...")
    
    # Start GPU monitoring
    gpu_monitor.start_monitoring(interval=0.5)
    
    sessions = []
    
    # Run 3 different training configurations
    configs = [
        {"epochs": 3, "batch_size": 64, "learning_rate": 0.001},
        {"epochs": 2, "batch_size": 128, "learning_rate": 0.01},
        {"epochs": 4, "batch_size": 32, "learning_rate": 0.0005}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Starting Training Run {i+1}/3")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        try:
            session_id, model = train_monitored_model(**config)
            sessions.append(session_id)
            print(f"Completed session: {session_id}")
        except Exception as e:
            print(f"Failed training run {i+1}: {e}")
        
        # Brief pause between runs
        time.sleep(2)
    
    return sessions

def print_monitoring_summary():
    """Print a summary of all monitoring data"""
    print("\n" + "="*80)
    print("MONITORING SUMMARY")
    print("="*80)
    
    # Active sessions
    active_sessions = training_monitor.get_active_sessions()
    print(f"Active Sessions: {len(active_sessions)}")
    
    for session_id, session in active_sessions.items():
        summary = training_monitor.get_session_summary(session_id)
        print(f"\nSession: {session_id}")
        print(f"  Model: {session.model_name}")
        print(f"  Status: {session.status}")
        print(f"  Duration: {summary.get('duration_minutes', 0):.1f} min")
        print(f"  Total Metrics: {summary.get('total_metrics', 0)}")
        
        # Latest metrics
        latest = training_monitor.get_latest_metrics(session_id)
        for metric_name, metric in latest.items():
            print(f"  Latest {metric_name}: {metric.value:.4f}")
    
    # GPU info
    gpu_metrics = gpu_monitor.get_latest_metrics()
    system_info = gpu_monitor.get_system_info()
    
    print(f"\nGPU Status:")
    print(f"  Simulated: {system_info.get('gpu_simulated', False)}")
    print(f"  PyTorch CUDA: {system_info.get('torch_available', False)}")
    
    for gpu in gpu_metrics:
        print(f"  GPU {gpu.gpu_id} ({gpu.name}):")
        print(f"    Utilization: {gpu.utilization:.1f}%")
        print(f"    Memory: {gpu.memory_used/1024:.1f}GB / {gpu.memory_total/1024:.1f}GB")
        print(f"    Temperature: {gpu.temperature:.1f}°C")
        print(f"    Power: {gpu.power_usage:.1f}W")

if __name__ == "__main__":
    print("Mini OS for Machine Learning and Training Databases - ML Training Monitor Demo")
    print("=" * 50)
    
    print("\nThis demo will:")
    print("1. Train multiple CNN models on MNIST")
    print("2. Collect real-time training metrics")
    print("3. Monitor GPU usage (real or simulated)")
    print("4. Generate visualizations")
    print("5. Provide WebSocket API for frontend")
    
    choice = input("\nRun demo? (y/n): ").lower().strip()
    
    if choice == 'y':
        try:
            # Run the demo
            sessions = simulate_multiple_training_runs()
            
            # Print summary
            print_monitoring_summary()
            
            print("\n" + "="*80)
            print("DEMO COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("\nTo access monitoring data:")
            print("1. Start the FastAPI server: uvicorn main:app --reload")
            print("2. Visit: http://localhost:8000/docs for API documentation")
            print("3. Use WebSocket: ws://localhost:8000/ws/monitoring")
            print("4. Check training logs in: ./training_logs/")
            
            print(f"\nCompleted Sessions: {sessions}")
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"\nDemo failed: {e}")
        finally:
            # Cleanup
            gpu_monitor.stop_monitoring()
            print("\nMonitoring stopped.")
    else:
        print("Demo cancelled.")
