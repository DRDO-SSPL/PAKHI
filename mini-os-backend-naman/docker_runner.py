import docker
import os

client = docker.from_env()

def run_user_code(code_path: str) -> str:
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
