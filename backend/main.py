# TODO: implement
from fastapi import FastAPI

app = FastAPI(title="AI-Based Network Packet Routing Optimization")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a minimal health response until the simulator is wired in."""
    return {"status": "ok"}
