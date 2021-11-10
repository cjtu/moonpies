try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # Python < 3.8
    import importlib_metadata
__version__ = importlib_metadata.version('moonpies')