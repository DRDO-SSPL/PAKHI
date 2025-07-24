import os 
import subprocess
import sys

try:
    import docker
    client = docker.from_env()
    DOCKER_AVAILABLE = True
    print("Docker client initialized successfully")
except Exception as e:
    DOCKER_AVAILABLE = False
    client = None
    print(f"Docker not available: {e}")

def run_user_code(code_path: str) -> str:
    """Run user code either in Docker container or directly"""
    
    if DOCKER_AVAILABLE and client:
        try:
            # Try to run in Docker container
            container = client.containers.run(
                image="mini-os-ml-image",
                command=f"python /app/code/{os.path.basename(code_path)}",
                volumes={
                    os.path.abspath(os.path.dirname(code_path)): {'bind': '/app/code', 'mode': 'rw'},
                },
                cpus=1,
                mem_limit="2g",
                auto_remove=True,
                detach=True
            )
            return container.logs().decode()
        except Exception as e:
            print(f"Docker execution failed: {e}")
            print("Falling back to direct execution...")
    
    # Fallback: run directly on host system
    print(f"Running code directly: {code_path}")
    try:
        result = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            cwd=os.path.dirname(code_path)
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n" + "=" * 50 + "\n"
            output += "STDERR:\n" + result.stderr
        
        if result.returncode != 0:
            output += f"\n\nProcess exited with code: {result.returncode}"
        
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (5 minutes limit)"
    except Exception as e:
        return f"Error executing code: {str(e)}"
