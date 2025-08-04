from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings
import secrets

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify the API key from the Authorization header.
    Only allows the specific hardcoded bearer token.
    """
    # Expected token (hardcoded for security)
    expected_token = "f187d1bc4df8a6a7e6cba86fc31bdedfcce699eac885b85570bb61c6d6e8c7f2"
    
    # Verify token matches exactly using constant-time comparison to prevent timing attacks
    if not credentials.credentials or not secrets.compare_digest(credentials.credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def get_api_key_header() -> str:
    """
    Get the expected API key for requests.
    """
    return settings.api_key