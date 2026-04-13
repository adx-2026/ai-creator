"""
AI Creator Space - Application entry point.

Assembles FastAPI app from modular route, provider, and queue components.
"""

import os
import asyncio

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.job_queue import job_queue, process_queue
from app.routes.auth_routes import router as auth_router
from app.routes.credit_routes import router as credit_router
from app.routes.model_routes import router as model_router
from app.routes.template_routes import router as template_router
from app.routes.settings_routes import router as settings_router
from app.routes.job_routes import router as job_router
from app.routes.file_routes import router as file_router
from app.routes.pipeline_routes import router as pipeline_router

# ------------------------------------------------------------------
# Create app and ensure directories
# ------------------------------------------------------------------
app = FastAPI()
os.makedirs("users", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ------------------------------------------------------------------
# Register all routers
# ------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(credit_router)
app.include_router(model_router)
app.include_router(template_router)
app.include_router(settings_router)
app.include_router(job_router)
app.include_router(file_router)
app.include_router(pipeline_router)


# ------------------------------------------------------------------
# Startup: load persisted jobs and launch background queue processor
# ------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    job_queue.load_jobs()
    asyncio.create_task(process_queue())


# ------------------------------------------------------------------
# Static files (must be last - catch-all mount)
# ------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
