from fastapi import APIRouter

router = APIRouter()


@router.get("/healthcheck")
def healthcheck():
    """Simple healthcheck endpoint"""
    return {"status": "ok"}
