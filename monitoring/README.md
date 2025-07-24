# Mini OS for Machine Learning and Training Databases - Monitoring System

The monitoring system in this project is designed to track and visualize metrics for machine learning training sessions in real-time. It includes GPU and training session monitoring, and provides an API for integration with frontend applications.

## Features

- **Training Session Monitoring**: Real-time tracking of training metrics such as loss, accuracy, and more.
- **GPU Monitoring**: Tracks GPU utilization, memory usage, temperature, and power consumption.
- **Visualizations**: Generate interactive charts and dashboards for better insights.
- **WebSocket API**: Provides real-time updates to frontend clients.

## Components

- `training_monitor.py`: Manages and logs training sessions and metrics.
- `gpu_monitor.py`: Monitors GPU metrics, with simulation for non-GPU environments.
- `visualization.py`: Creates plots for visualizing training and GPU metrics.
- `example_monitored_training.py`: Example script demonstrating integrated monitoring with a sample CNN training run.

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) For minimal installation:
   ```
   pip install -r requirements-minimal.txt
   ```

## Usage

### Training Monitoring

- Implement monitoring in your training scripts using `training_monitor`.
- Example usage is provided in `example_monitored_training.py`.

### Starting GPU Monitoring

- Utilize `gpu_monitor.py` to start tracking your GPU’s status.
- GPU simulation is available if no GPU is detected.

### Visualization

- Generate charts using `visualization.py`.
- Access plots such as Loss, Accuracy, and GPU stats via the visualization API.

## Example

Run `example_monitored_training.py` to see an example of the monitoring system in action:
```bash
python example_monitored_training.py
```

## API

- The monitoring API allows access to real-time data over WebSocket and includes endpoints for retrieving historical metrics.
- To explore API endpoints, start the server with FastAPI and visit the documentation at `http://localhost:8000/docs`.

## Contributing

Contributions are welcome! Please refer to the contribution guidelines in the main README file.

