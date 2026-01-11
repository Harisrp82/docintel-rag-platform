import pytest
from pathlib import Path
from loguru import logger

@pytest.fixture
def test_dirs():
    """Define test directories"""
    base_dir = Path(__file__).parent.parent
    return {
        "Vector DB": base_dir / "data" / "chroma_db",
        "Uploads": base_dir / "uploads"
    }

def test_directory_existence(test_dirs):
    """Test if required directories exist"""
    for name, path in test_dirs.items():
        assert path.exists(), f"{name} directory does not exist at: {path}"

def test_directory_permissions(test_dirs):
    """Test if directories are accessible and writable"""
    for name, path in test_dirs.items():
        try:
            test_file = path / "test_permissions.txt"
            test_file.touch()
            test_file.unlink()
            logger.info(f"✅ {name} directory is accessible at: {path.resolve()}")
        except Exception as e:
            pytest.fail(f"Failed to write to {name} directory: {str(e)}")