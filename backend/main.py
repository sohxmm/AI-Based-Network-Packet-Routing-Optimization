from fastapi import FastAPI

from api.routes import router as api_router

app = FastAPI(title="AI-Based Network Packet Routing Optimization")
app.include_router(api_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a minimal health response for uptime checks."""
    return {"status": "ok"}
