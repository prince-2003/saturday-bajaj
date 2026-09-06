from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from app.core.config import settings
import secrets

bearer_security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_security),
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Verify the API key from either 'Authorization: Bearer <token>' 
    or 'X-API-Key: <token>' header using constant-time comparison.
    """
    token = None
    if bearer and bearer.credentials:
        token = bearer.credentials
    elif api_key:
        token = api_key

    expected_token = settings.api_key or "f187d1bc4df8a6a7e6cba86fc31bdedfcce699eac885b85570bb61c6d6e8c7f2"

    if not token or not secrets.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

def get_api_key_header() -> str:
    """
    Get the expected API key for requests.
    """
    return settings.api_key