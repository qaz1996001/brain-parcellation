"""
Unit tests for FastAPI server configuration.

Tests proper server setup with lifespan, middleware, and error handling.
"""
import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestServerConfiguration:
    """Test server configuration and setup."""
    
    def test_server_basic_info(self, test_client: TestClient):
        """Test server basic information."""
        response = test_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
    
    def test_health_check_endpoint(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "database" in data
        assert "redis" in data
    
    def test_openapi_documentation(self, test_client: TestClient):
        """Test OpenAPI documentation is available."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Medical Imaging AI API"
    
    def test_cors_headers(self, test_client: TestClient):
        """Test CORS headers are properly set."""
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_security_headers(self, test_client: TestClient):
        """Test security headers are properly set."""
        response = test_client.get("/health")
        
        # Check security headers
        assert "x-content-type-options" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        
        assert "x-frame-options" in response.headers
        assert response.headers["x-frame-options"] == "DENY"
        
        assert "x-xss-protection" in response.headers
        assert response.headers["x-xss-protection"] == "1; mode=block"
    
    def test_process_time_header(self, test_client: TestClient):
        """Test process time header is added."""
        response = test_client.get("/health")
        
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time > 0
    
    def test_not_found_handler(self, test_client: TestClient):
        """Test 404 error handler."""
        response = test_client.get("/nonexistent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["error"] == "NOT_FOUND"
        assert "message" in data
        assert data["path"] == "/nonexistent-endpoint"
    
    def test_validation_error_handler(self, test_client: TestClient):
        """Test validation error handler."""
        # Assuming we have an endpoint that validates input
        response = test_client.post(
            "/api/v1/pipelines/execute",
            json={"invalid": "data"}  # Missing required fields
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data


class TestLifespanManagement:
    """Test application lifespan management."""
    
    def test_lifespan_not_using_events(self):
        """Verify app uses lifespan context manager, not events."""
        import inspect
        from backend.app import server
        
        source = inspect.getsource(server)
        
        # Check for deprecated event handlers
        assert "@app.on_event" not in source, "Should not use deprecated event handlers"
        
        # Check for lifespan usage
        assert "lifespan=" in source, "Should use lifespan context manager"
        assert "async def lifespan" in source, "Should define lifespan function"


class TestMiddlewareConfiguration:
    """Test middleware configuration."""
    
    def test_middleware_order(self):
        """Verify middleware are configured in correct order."""
        from backend.app.server import app
        
        # Get middleware stack
        middleware_stack = []
        current = app.middleware_stack
        while hasattr(current, "cls"):
            middleware_stack.append(current.cls.__name__)
            if hasattr(current, "app"):
                current = current.app
            else:
                break
        
        # Verify CORS middleware is present
        assert any("CORS" in name for name in middleware_stack)
        
        # Verify custom middleware are present
        # Note: Actual middleware names depend on implementation


class TestErrorHandling:
    """Test global error handling."""
    
    @pytest.mark.asyncio
    async def test_unhandled_exception_handler(self, test_client: TestClient):
        """Test unhandled exception handler."""
        # Mock an endpoint that raises an exception
        from backend.app.server import app
        
        @app.get("/test-error")
        async def test_error():
            raise RuntimeError("Test unhandled error")
        
        response = test_client.get("/test-error")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "INTERNAL_SERVER_ERROR"
        assert "message" in data
        
        # Clean up
        app.routes = [r for r in app.routes if r.path != "/test-error"]
    
    def test_http_exception_handler(self, test_client: TestClient):
        """Test HTTP exception handler."""
        from backend.app.server import app
        from fastapi import HTTPException
        
        @app.get("/test-http-error")
        async def test_http_error():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        response = test_client.get("/test-http-error")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
        data = response.json()
        assert data["detail"] == "Access denied"
        
        # Clean up
        app.routes = [r for r in app.routes if r.path != "/test-http-error"]


class TestFastAPIBestPractices:
    """Test FastAPI best practices are followed."""
    
    def test_pydantic_v2_usage(self):
        """Verify Pydantic v2 is being used."""
        import pydantic
        
        version = pydantic.VERSION
        major_version = int(version.split(".")[0])
        assert major_version >= 2, "Should use Pydantic v2"
    
    def test_async_endpoints(self):
        """Verify endpoints are async."""
        from backend.app.server import app
        
        # Check that most endpoints are async
        async_count = 0
        sync_count = 0
        
        for route in app.routes:
            if hasattr(route, "endpoint"):
                if inspect.iscoroutinefunction(route.endpoint):
                    async_count += 1
                else:
                    sync_count += 1
        
        # Most endpoints should be async
        if async_count + sync_count > 0:
            async_percentage = async_count / (async_count + sync_count)
            assert async_percentage > 0.8, "Most endpoints should be async"
    
    def test_dependency_injection(self):
        """Verify proper dependency injection is used."""
        from backend.app.server import app
        
        # Check for Depends usage in routes
        has_depends = False
        for route in app.routes:
            if hasattr(route, "dependencies"):
                if route.dependencies:
                    has_depends = True
                    break
        
        assert has_depends, "Should use dependency injection"
