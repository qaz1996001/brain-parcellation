# app/main.py
import os
import uvicorn
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    #
    AI_APP_PORT = int(os.getenv("AI_APP_PORT",8000))
    uvicorn.run("backend.app.server:app", host="0.0.0.0",
                port=AI_APP_PORT, reload=False)
