from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import endpoints
from app.services.gpu_diagnostics import gpu_diagnostics
from app.services.job_worker import worker


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker.start()
    gpu_diagnostics.start_check()
    yield
    worker.stop()


app = FastAPI(title="Reference-Based Super-Resolution API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(endpoints.router, prefix="/api/v1")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Reference-Based Super-Resolution API"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
