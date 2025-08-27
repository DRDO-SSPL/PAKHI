from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil, os, asyncio, threading, uuid, sys
from datetime import datetime
from typing import Dict, List, Optional
from docker_runner import run_user_code

# Import monitoring components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'monitoring'))
try:
    from training_monitor import training_monitor, MonitoredModel
    from gpu_monitor import gpu_monitor
    from visualization import MetricsVisualizer
    from websocket_api import websocket_server
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Monitoring system not available: {e}")
    MONITORING_AVAILABLE = False

app = FastAPI(title="Mini OS for Machine Learning and Training Databases - ML Training Platform", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize monitoring components
if MONITORING_AVAILABLE:
    visualizer = MetricsVisualizer(training_monitor, gpu_monitor)
else:
    visualizer = None

# Start GPU monitoring on startup
@app.on_event("startup")
async def startup_event():
    """Initialize monitoring systems"""
    if MONITORING_AVAILABLE:
        gpu_monitor.start_monitoring(interval=1.0)
        print("ML Training Monitoring System initialized")
    else:
        print("Monitoring system not available")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup monitoring systems"""
    if MONITORING_AVAILABLE:
        gpu_monitor.stop_monitoring()
        print("ML Training Monitoring System stopped")

@app.post("/upload/")
async def upload_and_run(file: UploadFile = File(...)):
    """Upload and run user code with monitoring"""
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Create a unique session for this execution
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        model_name = f"model_{file.filename.split('.')[0]}"
        
        # Start monitoring session
        training_monitor.start_session(session_id, model_name)
        
        result = run_user_code(file_location)
        
        # End monitoring session
        training_monitor.end_session(session_id, "completed")
        
        return {
            "status": "success", 
            "output": result,
            "session_id": session_id,
            "monitoring_url": f"/monitoring/session/{session_id}"
        }
    except Exception as e:
        if 'session_id' in locals():
            training_monitor.end_session(session_id, "failed")
        return {"status": "error", "error": str(e)}

# ============================================================================
# MONITORING API ENDPOINTS
# ============================================================================

@app.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    if not MONITORING_AVAILABLE:
        return {"status": "unavailable", "message": "Monitoring system not installed"}
    
    system_info = gpu_monitor.get_system_info()
    active_sessions = training_monitor.get_active_sessions()
    
    return {
        "status": "active",
        "system_info": system_info,
        "active_sessions": len(active_sessions),
        "gpu_monitoring": gpu_monitor.monitoring,
        "websocket_url": "ws://localhost:8765"
    }

@app.get("/monitoring/sessions")
async def list_training_sessions():
    """Get list of all training sessions"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    active_sessions = training_monitor.get_active_sessions()
    sessions_data = []
    
    for session_id, session in active_sessions.items():
        summary = training_monitor.get_session_summary(session_id)
        sessions_data.append({
            "session_id": session_id,
            "model_name": session.model_name,
            "start_time": session.start_time.isoformat(),
            "status": session.status,
            "summary": summary
        })
    
    return {"sessions": sessions_data}

@app.get("/monitoring/session/{session_id}")
async def get_session_data(session_id: str):
    """Get detailed data for a specific training session"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    session_data = visualizer.create_plotly_dashboard_data(session_id)
    if "error" in session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return session_data

@app.get("/monitoring/session/{session_id}/metrics/{metric_name}")
async def get_session_metric(session_id: str, metric_name: str, last_n: int = 100):
    """Get specific metric history for a session"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    history = training_monitor.get_metric_history(session_id, metric_name, last_n)
    
    if not history:
        raise HTTPException(status_code=404, detail=f"No {metric_name} data for session {session_id}")
    
    return {
        "session_id": session_id,
        "metric_name": metric_name,
        "data": [
            {
                "epoch": m.epoch,
                "step": m.step,
                "value": m.value,
                "timestamp": m.timestamp.isoformat()
            }
            for m in history
        ]
    }

@app.get("/monitoring/gpu")
async def get_gpu_status():
    """Get current GPU status and metrics"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    gpu_metrics = gpu_monitor.get_latest_metrics()
    system_info = gpu_monitor.get_system_info()
    
    return {
        "gpu_metrics": [
            {
                "gpu_id": gpu.gpu_id,
                "name": gpu.name,
                "utilization": gpu.utilization,
                "memory_used": gpu.memory_used,
                "memory_total": gpu.memory_total,
                "memory_percent": gpu.memory_percent,
                "temperature": gpu.temperature,
                "power_usage": gpu.power_usage,
                "timestamp": gpu.timestamp.isoformat()
            }
            for gpu in gpu_metrics
        ],
        "system_info": system_info
    }

@app.get("/monitoring/charts/training/{session_id}")
async def get_training_chart(session_id: str):
    """Get training dashboard chart as base64 image"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    try:
        chart_image = visualizer.create_training_dashboard(session_id)
        return {"image": chart_image, "format": "png", "encoding": "base64"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@app.get("/monitoring/charts/gpu")
async def get_gpu_chart():
    """Get GPU monitoring chart as base64 image"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    try:
        chart_image = visualizer.create_gpu_dashboard()
        return {"image": chart_image, "format": "png", "encoding": "base64"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@app.get("/monitoring/charts/metric/{session_id}/{metric_name}")
async def get_metric_chart(session_id: str, metric_name: str, last_n: int = 100):
    """Get specific metric chart as base64 image"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    try:
        chart_image = visualizer.create_live_metric_chart(session_id, metric_name, last_n)
        return {"image": chart_image, "format": "png", "encoding": "base64"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

# WebSocket endpoint for real-time monitoring
@app.websocket("/ws/monitoring")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates"""
    if not MONITORING_AVAILABLE:
        await websocket.close(code=1011, reason="Monitoring not available")
        return
    
    await websocket.accept()
    client_id = id(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Mini OS for Machine Learning and Training Databases - ML Training Monitor",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Wait for client messages
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "subscribe_training":
                    session_id = data.get("session_id", "all")
                    # Send initial training data
                    if session_id != "all":
                        session_data = visualizer.create_plotly_dashboard_data(session_id)
                        await websocket.send_json({
                            "type": "training_data",
                            "session_id": session_id,
                            "data": session_data
                        })
                
                elif message_type == "subscribe_gpu":
                    # Send initial GPU data
                    gpu_metrics = gpu_monitor.get_latest_metrics()
                    system_info = gpu_monitor.get_system_info()
                    await websocket.send_json({
                        "type": "gpu_data",
                        "data": {
                            "gpu_metrics": [
                                {
                                    "gpu_id": gpu.gpu_id,
                                    "name": gpu.name,
                                    "utilization": gpu.utilization,
                                    "memory_used": gpu.memory_used,
                                    "memory_total": gpu.memory_total,
                                    "memory_percent": gpu.memory_percent,
                                    "temperature": gpu.temperature,
                                    "power_usage": gpu.power_usage,
                                    "timestamp": gpu.timestamp.isoformat()
                                }
                                for gpu in gpu_metrics
                            ],
                            "system_info": system_info
                        }
                    })
                
                elif message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except asyncio.TimeoutError:
                # Send periodic updates
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
        await websocket.close(code=1011, reason=str(e))

# ============================================================================
# SERVER STARTUP INFORMATION
# ============================================================================

# to start the FastAPI server, run:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

# For development with monitoring:
# pip install psutil matplotlib seaborn pandas websockets

# To build the Docker Image
# docker build -t mini-os-ml-image .
