#!/usr/bin/env python3
"""
Debug script to test imports one by one to identify the problematic module.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module and report the result."""
    try:
        print(f"Testing {description}...")
        if module_name == "app.core.logging":
            from app.core.logging import setup_logging
            setup_logging()
            print(f"✅ {description} imported successfully")
        elif module_name == "app.core.config":
            from app.core.config import settings
            print(f"✅ {description} imported successfully")
        elif module_name == "app.models.schemas":
            from app.models.schemas import QueryRequest, QueryResponse, EmbeddingChunk
            print(f"✅ {description} imported successfully")
        elif module_name == "app.core.container":
            from app.core.container import get_service_container
            print(f"✅ {description} imported successfully")
        elif module_name == "app.api.endpoints":
            from app.api.endpoints import hackrx, health
            print(f"✅ {description} imported successfully")
        else:
            __import__(module_name)
            print(f"✅ {description} imported successfully")
    except Exception as e:
        print(f"❌ {description} failed to import: {str(e)}")
        traceback.print_exc()
        return False
    return True

def main():
    print("=== Testing Critical Imports ===\n")
    
    # Test basic Python packages
    test_import("structlog", "Structured logging")
    test_import("fastapi", "FastAPI framework")
    test_import("pydantic", "Pydantic validation")
    test_import("uvicorn", "Uvicorn ASGI server")
    
    print("\n=== Testing Application Modules ===\n")
    
    # Test app modules in order
    test_import("app.core.logging", "App logging setup")
    test_import("app.core.config", "App configuration")
    test_import("app.models.schemas", "App schemas")
    test_import("app.core.container", "Service container")
    test_import("app.api.endpoints", "API endpoints")
    
    print("\n=== Testing Service Dependencies ===\n")
    
    # Test service dependencies
    test_import("openai", "OpenAI API client")
    test_import("qdrant_client", "Qdrant vector database")
    test_import("redis", "Redis caching")
    test_import("PyPDF2", "PDF processing")
    test_import("pdfplumber", "Advanced PDF processing")
    test_import("psutil", "System monitoring")

if __name__ == "__main__":
    main()
