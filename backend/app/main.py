# app/main.py
import os
import uvicorn
from code_ai import load_dotenv



if __name__ == "__main__":
    from backend.app.server import app
    load_dotenv()
    #
    APP_PORT = int(os.getenv("APP_PORT",8000))
    uvicorn.run("backend.app.server:app", host="0.0.0.0",
                port=APP_PORT, reload=False)
