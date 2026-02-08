from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.api import endpoints

app = FastAPI(title="Reference-Based Super-Resolution API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
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
