from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import os
from datetime import datetime
import cv2

# Import your LLM functions
from use_LLM import get_LLM_response
from use_STT import get_STT_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eneles.ai", "https://155c-157-157-221-29.ngrok-free.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = "uploaded_audio.wav"
    file_path = os.path.join(os.getcwd(), filename)
    
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
        
    file_size = os.path.getsize(file_path)
    modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))

    # Log the details
    print(f"Received file: {file.filename}")
    print(f"File saved as: {file_path}")
    print(f"File size: {file_size} bytes")
    print(f"Last modified: {modified_time}")

    # Process the audio file with your LLM
    input_text = get_STT_response(file_path)
    response_text = get_LLM_response("Get the response", input_text)

    return {
        "message": "File uploaded and processed successfully",
        "response": response_text
    }

def generate_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail="Video file not found")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Send a special signal indicating the end of the video
    yield (b'--frame\r\n'
           b'Content-Type: text/plain\r\n\r\n' + b'END_OF_VIDEO' + b'\r\n')

@app.get("/stream-video/")
async def stream_video():
    video_path = "test.mp4"  # Replace with the actual path to your video file
    return StreamingResponse(generate_video_stream(video_path), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
