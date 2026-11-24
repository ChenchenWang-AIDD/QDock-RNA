"""QDock manuscript docking core (solver dependencies are loaded lazily)."""

__all__ = ["GPMDock"]


def __getattr__(name):
    if name == "GPMDock":
        from .qdock import GPMDock
        return GPMDock
    raise AttributeError(name)
