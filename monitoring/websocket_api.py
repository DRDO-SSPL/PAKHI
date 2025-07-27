"""
WebSocket API for Real-time ML Training Monitoring
Provides live updates to frontend via WebSocket connections
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime
import threading
from dataclasses import asdict

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Warning: websockets not available. Install with: pip install websockets")

from training_monitor import TrainingMonitor, TrainingMetric, training_monitor
from gpu_monitor import GPUMonitor, gpu_monitor
from visualization import MetricsVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringWebSocketServer:
    """WebSocket server for real-time ML monitoring updates"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        self.server = None
        
        # Monitoring components
        self.training_monitor = training_monitor
        self.gpu_monitor = gpu_monitor
        self.visualizer = MetricsVisualizer(self.training_monitor, self.gpu_monitor)
        
        # Message handlers
        self.message_handlers = {
            'subscribe_training': self._handle_subscribe_training,
            'subscribe_gpu': self._handle_subscribe_gpu,
            'get_session_data': self._handle_get_session_data,
            'get_gpu_data': self._handle_get_gpu_data,
            'get_chart_data': self._handle_get_chart_data,
            'list_sessions': self._handle_list_sessions,
            'ping': self._handle_ping
        }
        
        # Subscription tracking
        self.training_subscriptions: Dict[WebSocketServerProtocol, Set[str]] = {}
        self.gpu_subscriptions: Set[WebSocketServerProtocol] = set()
        
        # Update intervals (seconds)
        self.training_update_interval = 2.0
        self.gpu_update_interval = 1.0
        
        # Setup metric callbacks
        self.training_monitor.add_callback(self._on_new_training_metric)
    
    async def start_server(self):
        """Start the WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets package not available")
        
        self.running = True
        
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring(interval=1.0)
        
        # Start update tasks
        asyncio.create_task(self._training_broadcast_loop())
        asyncio.create_task(self._gpu_broadcast_loop())
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"Monitoring WebSocket server started on ws://{self.host}:{self.port}")
        
        # Keep server running
        await self.server.wait_closed()
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        self.running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Stop GPU monitoring
        self.gpu_monitor.stop_monitoring()
        
        logger.info("Monitoring WebSocket server stopped")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection"""
        self.clients.add(websocket)
        logger.info(f"New client connected: {websocket.remote_address}")
        
        try:
            # Send welcome message
            await self._send_message(websocket, {
                'type': 'welcome',
                'message': 'Connected to ML Training Monitor',
                'timestamp': datetime.now().isoformat()
            })
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self._send_error(websocket, f"Error processing message: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        
        finally:
            # Clean up client
            self.clients.discard(websocket)
            self.training_subscriptions.pop(websocket, None)
            self.gpu_subscriptions.discard(websocket)
    
    async def _handle_message(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle incoming WebSocket message"""
        message_type = data.get('type')
        
        if message_type in self.message_handlers:
            await self.message_handlers[message_type](websocket, data)
        else:
            await self._send_error(websocket, f"Unknown message type: {message_type}")
    
    async def _handle_subscribe_training(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle training metrics subscription"""
        session_id = data.get('session_id', 'all')
        
        if websocket not in self.training_subscriptions:
            self.training_subscriptions[websocket] = set()
        
        self.training_subscriptions[websocket].add(session_id)
        
        await self._send_message(websocket, {
            'type': 'subscription_confirmed',
            'subscription_type': 'training',
            'session_id': session_id,
            'message': f'Subscribed to training metrics for session: {session_id}'
        })
        
        # Send initial data
        if session_id != 'all':
            session_data = self.visualizer.create_plotly_dashboard_data(session_id)
            await self._send_message(websocket, {
                'type': 'training_data',
                'session_id': session_id,
                'data': session_data
            })
    
    async def _handle_subscribe_gpu(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle GPU metrics subscription"""
        self.gpu_subscriptions.add(websocket)
        
        await self._send_message(websocket, {
            'type': 'subscription_confirmed',
            'subscription_type': 'gpu',
            'message': 'Subscribed to GPU metrics'
        })
        
        # Send initial GPU data
        gpu_data = self._get_gpu_data()
        await self._send_message(websocket, {
            'type': 'gpu_data',
            'data': gpu_data
        })
    
    async def _handle_get_session_data(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle request for session data"""
        session_id = data.get('session_id')
        
        if not session_id:
            await self._send_error(websocket, "session_id required")
            return
        
        session_data = self.visualizer.create_plotly_dashboard_data(session_id)
        await self._send_message(websocket, {
            'type': 'session_data',
            'session_id': session_id,
            'data': session_data
        })
    
    async def _handle_get_gpu_data(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle request for GPU data"""
        gpu_data = self._get_gpu_data()
        await self._send_message(websocket, {
            'type': 'gpu_data',
            'data': gpu_data
        })
    
    async def _handle_get_chart_data(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle request for chart data"""
        chart_type = data.get('chart_type')
        session_id = data.get('session_id')
        
        if chart_type == 'training_dashboard' and session_id:
            chart_data = self.visualizer.create_training_dashboard(session_id)
            await self._send_message(websocket, {
                'type': 'chart_data',
                'chart_type': chart_type,
                'session_id': session_id,
                'image': chart_data
            })
        
        elif chart_type == 'gpu_dashboard':
            chart_data = self.visualizer.create_gpu_dashboard()
            await self._send_message(websocket, {
                'type': 'chart_data',
                'chart_type': chart_type,
                'image': chart_data
            })
        
        else:
            await self._send_error(websocket, "Invalid chart_type or missing session_id")
    
    async def _handle_list_sessions(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle request for active sessions list"""
        active_sessions = self.training_monitor.get_active_sessions()
        
        sessions_data = []
        for session_id, session in active_sessions.items():
            summary = self.training_monitor.get_session_summary(session_id)
            sessions_data.append({
                'session_id': session_id,
                'model_name': session.model_name,
                'start_time': session.start_time.isoformat(),
                'status': session.status,
                'summary': summary
            })
        
        await self._send_message(websocket, {
            'type': 'sessions_list',
            'sessions': sessions_data
        })
    
    async def _handle_ping(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle ping message"""
        await self._send_message(websocket, {
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        })
    
    async def _training_broadcast_loop(self):
        """Broadcast training updates to subscribed clients"""
        while self.running:
            try:
                if self.training_subscriptions:
                    # Get all active sessions
                    active_sessions = self.training_monitor.get_active_sessions()
                    
                    for websocket, session_ids in self.training_subscriptions.items():
                        if websocket.closed:
                            continue
                        
                        for session_id in session_ids:
                            if session_id == 'all':
                                # Send data for all active sessions
                                for active_session_id in active_sessions.keys():
                                    session_data = self.visualizer.create_plotly_dashboard_data(active_session_id)
                                    await self._send_message(websocket, {
                                        'type': 'training_update',
                                        'session_id': active_session_id,
                                        'data': session_data
                                    })
                            elif session_id in active_sessions:
                                # Send data for specific session
                                session_data = self.visualizer.create_plotly_dashboard_data(session_id)
                                await self._send_message(websocket, {
                                    'type': 'training_update',
                                    'session_id': session_id,
                                    'data': session_data
                                })
                
                await asyncio.sleep(self.training_update_interval)
                
            except Exception as e:
                logger.error(f"Error in training broadcast loop: {e}")
                await asyncio.sleep(self.training_update_interval)
    
    async def _gpu_broadcast_loop(self):
        """Broadcast GPU updates to subscribed clients"""
        while self.running:
            try:
                if self.gpu_subscriptions:
                    gpu_data = self._get_gpu_data()
                    
                    # Send to all GPU subscribers
                    disconnected_clients = set()
                    for websocket in self.gpu_subscriptions:
                        if websocket.closed:
                            disconnected_clients.add(websocket)
                            continue
                        
                        await self._send_message(websocket, {
                            'type': 'gpu_update',
                            'data': gpu_data
                        })
                    
                    # Clean up disconnected clients
                    self.gpu_subscriptions -= disconnected_clients
                
                await asyncio.sleep(self.gpu_update_interval)
                
            except Exception as e:
                logger.error(f"Error in GPU broadcast loop: {e}")
                await asyncio.sleep(self.gpu_update_interval)
    
    def _on_new_training_metric(self, metric: TrainingMetric):
        """Callback for new training metrics"""
        # This will be handled by the broadcast loop
        # You could add immediate notification here if needed
        pass
    
    def _get_gpu_data(self) -> Dict:
        """Get current GPU data"""
        gpu_metrics = self.gpu_monitor.get_latest_metrics()
        system_info = self.gpu_monitor.get_system_info()
        
        return {
            'gpu_metrics': [
                {
                    'gpu_id': gpu.gpu_id,
                    'name': gpu.name,
                    'utilization': gpu.utilization,
                    'memory_used': gpu.memory_used,
                    'memory_total': gpu.memory_total,
                    'memory_percent': gpu.memory_percent,
                    'temperature': gpu.temperature,
                    'power_usage': gpu.power_usage,
                    'timestamp': gpu.timestamp.isoformat()
                }
                for gpu in gpu_metrics
            ],
            'system_info': system_info,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _send_message(self, websocket: WebSocketServerProtocol, message: Dict):
        """Send message to WebSocket client"""
        try:
            if not websocket.closed:
                await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            # Client disconnected
            pass
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """Send error message to WebSocket client"""
        await self._send_message(websocket, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })

# Global WebSocket server instance
websocket_server = MonitoringWebSocketServer()

# Standalone server runner
async def run_monitoring_server(host: str = "localhost", port: int = 8765):
    """Run the monitoring WebSocket server"""
    server = MonitoringWebSocketServer(host, port)
    await server.start_server()

if __name__ == "__main__":
    # Run server directly
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    
    print(f"Starting ML Training Monitor WebSocket Server...")
    print(f"Server will run on ws://{host}:{port}")
    
    try:
        asyncio.run(run_monitoring_server(host, port))
    except KeyboardInterrupt:
        print("\\nServer stopped by user")
    except Exception as e:
        print(f"Error running server: {e}")
