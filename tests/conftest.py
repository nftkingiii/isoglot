import pytest


@pytest.fixture(autouse=True)
def isolated_auditkit_cache(tmp_path, monkeypatch):
    """AuditKIT caches run results under XDG_CACHE_HOME; give each test its own
    so results never leak between tests or test runs."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
