from fastapi import FastAPI

app = FastAPI(title="Harmonizer API", version="0.0.1")

@app.get("/")
async def root():
    """Health‑checexik endpoint for load balancers & CI."""
    return {"status": "ok"}
