import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run('main:app', host='127.0.0.1', port=port)
