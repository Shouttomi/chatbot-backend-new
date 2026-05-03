
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from app.auth.jwt import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):

    # Demo credentials
    if form_data.username != "admin" or form_data.password != "12345":
        return {"error": "Invalid credentials"}

    token = create_access_token({"sub": form_data.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }
