import pytest
from config import settings

def test_storage_paths():
    """Test if storage paths are correctly configured"""
    assert settings.VECTOR_DB_PATH.exists()
    assert settings.UPLOAD_DIR.exists()