"""
Crowd Monitoring API Server
============================
FastAPI application exposing crowd estimation and system health endpoints.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Or via Docker / docker-compose (see docker-compose.yml).

OpenAPI docs available at:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from crowd_engine.domain.entities import CameraInput
from crowd_engine.infra.config import settings
from crowd_engine.infra.logger import get_logger, set_correlation_id
from crowd_engine.services.detection_service import DetectionService
from crowd_engine.services.factory import build_orchestrator
from crowd_engine.services.health import HealthService
from crowd_engine.services.factory import build_providers

log = get_logger("api_server")

# ── Startup / shutdown ─────────────────────────────────────────────────────

_orchestrator = None
_health_service = None
_detection_service: Optional[DetectionService] = None


def _load_and_start_cameras(svc: DetectionService) -> None:
    """Read cameras.json and start a detection worker for every camera."""
    cameras_path = Path(settings.cameras_file)
    if not cameras_path.exists():
        log.warning("cameras.json not found — no cameras auto-started")
        return
    try:
        cameras = json.loads(cameras_path.read_text())
        for cam in cameras:
            cam_id = cam.get("camera_id") or str(uuid.uuid4())
            svc.start_camera(
                camera_id=cam_id,
                source=cam.get("source", 0),
                label=cam.get("label", cam_id),
                line_cfg=cam.get("counting_line"),
            )
    except Exception as exc:
        log.error("Failed to load cameras: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _health_service, _detection_service
    log.info("Starting Crowd Monitoring API")
    providers = build_providers()
    _orchestrator = build_orchestrator()
    _health_service = HealthService(providers)
    _detection_service = DetectionService()
    _load_and_start_cameras(_detection_service)
    yield
    log.info("Shutting down Crowd Monitoring API")
    if _detection_service:
        for cam in _detection_service.list_cameras():
            _detection_service.stop_camera(cam.camera_id)


# ── App factory ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Crowd Monitoring API",
    description=(
        "Modular, provider-agnostic crowd estimation API with automatic fallback. "
        "Supports multiple providers: Roboflow, HuggingFace, Geospatial, OpenCV."
    ),
    version="2.0.0",
    contact={"name": "Crowd Monitoring Team"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Rate limiter (in-memory, per IP) ──────────────────────────────────────

_rate_store: Dict[str, Dict[str, Any]] = {}
_RATE_WINDOW = 60  # seconds


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    record = _rate_store.setdefault(ip, {"count": 0, "window_start": now})
    if now - record["window_start"] > _RATE_WINDOW:
        record["count"] = 0
        record["window_start"] = now
    record["count"] += 1
    return record["count"] <= settings.rate_limit_per_minute


# ── Middleware ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    set_correlation_id(cid)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after_seconds": _RATE_WINDOW},
        )
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── Request / Response schemas ─────────────────────────────────────────────

class EstimateRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Camera latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Camera longitude")
    source: str = Field(..., description="Image/video source: URL, file path, or device index")
    camera_id: Optional[str] = Field(None, description="Optional stable camera identifier")
    label: Optional[str] = Field(None, description="Human-readable camera label")

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source must not be empty")
        return v


class EstimateResponse(BaseModel):
    count: int
    confidence: float
    timestamp: str
    source: str
    camera_id: str
    latitude: float
    longitude: float
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    """API root — returns version info."""
    return {"name": "Crowd Monitoring API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["observability"], summary="System health check")
def health_check():
    """
    Returns health status for all configured providers plus aggregate metrics.

    Overall status is `ok` if at least one provider is healthy, `degraded` otherwise.
    """
    if _health_service is None:
        raise HTTPException(status_code=503, detail="Service initialising")
    return _health_service.check()


@app.get("/readyz", tags=["observability"], summary="Kubernetes readiness probe")
def readyz():
    """Lightweight readiness probe — 200 when the app is ready to serve."""
    if _orchestrator is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return {"ready": True}


@app.get("/livez", tags=["observability"], summary="Kubernetes liveness probe")
def livez():
    """Lightweight liveness probe — always 200 if the process is alive."""
    return {"alive": True}


@app.post(
    "/api/v1/estimate",
    response_model=EstimateResponse,
    tags=["crowd"],
    summary="Estimate crowd count",
    status_code=status.HTTP_200_OK,
)
def estimate_crowd(body: EstimateRequest):
    """
    Estimate the crowd count for a given camera source.

    The orchestrator tries providers in order (as configured by PROVIDER_CHAIN):
    1. Roboflow API (if ROBOFLOW_API_KEY configured)
    2. HuggingFace local model / Inference API
    3. Geospatial / OSM heuristic (low confidence)
    4. OpenCV MobileNetSSD (legacy degraded mode)

    Returns the first valid estimate, or an error estimate if all fail.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service initialising")

    camera = CameraInput(
        source=body.source,
        latitude=body.latitude,
        longitude=body.longitude,
        camera_id=body.camera_id or str(uuid.uuid4()),
        label=body.label or body.source,
    )
    result = _orchestrator.estimate(camera)
    return result.as_dict()


@app.get(
    "/api/v1/orchestrator/health",
    tags=["observability"],
    summary="Orchestrator metrics",
)
def orchestrator_health():
    """
    Returns detailed orchestrator metrics: success rate, fallback rate,
    per-provider circuit-breaker status, and call counts.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service initialising")
    return _orchestrator.health()


# ── Live detection — streaming & camera management ─────────────────────────

@app.get("/stream/{camera_id}", tags=["streaming"], summary="MJPEG video stream")
def stream_camera(camera_id: str):
    """
    MJPEG video stream for a running camera.
    Display directly in an HTML <img> tag:
        <img src="http://localhost:8000/stream/kiu-main-entrance">
    """
    if _detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not ready")

    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            frame = _detection_service.get_latest_frame(camera_id)
            if frame:
                yield boundary + frame + b"\r\n"
            time.sleep(0.033)  # ~30 fps cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/cameras", tags=["detection"], summary="List all cameras")
def list_cameras():
    """Return stats for all configured cameras."""
    if _detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not ready")
    cameras = _detection_service.list_cameras()
    return [
        {
            "camera_id":     c.camera_id,
            "label":         c.label,
            "source":        c.source,
            "running":       c.running,
            "count_in":      c.count_in,
            "count_out":     c.count_out,
            "current_count": c.current_count,
            "fps":           c.fps,
            "session_start": c.session_start,
        }
        for c in cameras
    ]


@app.get("/api/v1/cameras/{camera_id}", tags=["detection"], summary="Camera stats")
def camera_stats(camera_id: str):
    """Return live stats for a single camera."""
    if _detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not ready")
    s = _detection_service.get_stats(camera_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id!r} not found")
    return {
        "camera_id":     s.camera_id,
        "label":         s.label,
        "source":        s.source,
        "running":       s.running,
        "count_in":      s.count_in,
        "count_out":     s.count_out,
        "current_count": s.current_count,
        "fps":           s.fps,
        "session_start": s.session_start,
    }


@app.post("/api/v1/cameras/{camera_id}/reset", tags=["detection"], summary="Reset counts")
def reset_camera_counts(camera_id: str):
    """Reset IN/OUT counters for a camera (keeps it running)."""
    if _detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not ready")
    ok = _detection_service.reset_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id!r} not found")
    return {"reset": True, "camera_id": camera_id}


@app.get("/api/v1/cameras/{camera_id}/events", tags=["detection"], summary="Crossing events")
def camera_events(camera_id: str, limit: int = 100):
    """Return the last N crossing events for a camera."""
    if _detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not ready")
    return _detection_service.get_events(camera_id=camera_id, limit=limit)


@app.get("/api/v1/metrics", tags=["detection"], summary="System-wide metrics")
def detection_metrics():
    """
    Aggregate metrics across all cameras.
    Useful for evaluation (total counts, average FPS, occupancy).
    """
    if _detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not ready")
    return _detection_service.get_metrics()


# ── Dev entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )

