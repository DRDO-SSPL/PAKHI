from fastapi import FastAPI, UploadFile, File
import shutil, os
from docker_runner import run_user_code

app = FastAPI()
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_and_run(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = run_user_code(file_location)
        return {"status": "success", "output": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# to start the FastAPI server, run:
# uvicorn main:app --reload


# To build the Docker Image
# docker build -t mini-os-ml-image .
