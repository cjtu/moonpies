# Plot helpers
import warnings
from pathlib import Path
import matplotlib as mpl

try:
    import default_config
except ModuleNotFoundError:
    from moonpies import default_config

CFG = default_config.Cfg(seed=1)
WARNINGS = 0  # Count warnings

# Plot helpers
def reset_plot_style(mplstyle=True, cfg=CFG):
    """Reset matplotlib style defaults, use MoonPIES mplstyle if True."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    if mplstyle:
        if mplstyle is True:
            mplstyle = Path(cfg.model_path) / ".moonpies.mplstyle"
        try:
            mpl.style.use(mplstyle)
        except (OSError, FileNotFoundError):
            global WARNINGS
            if WARNINGS < 1:
                warnings.warn(f"Could not find mplstyle file {mplstyle}")
                WARNINGS += 1


def plot_version(
    cfg=CFG, loc="ll", xy=None, xyoff=None, ax=None, bbkw=None, **kwargs
):
    """Add MoonPIES version label."""
    x, y = (0, 0) if xy is None else xy
    xoff, yoff = (0, 0) if xyoff is None else xyoff
    ax = mpl.pyplot.gca() if ax is None else ax
    bbkw = {} if bbkw is None else bbkw
    # Get position of version label
    if loc[0] == "l":  # lower
        y += 0.035 + yoff
        va = "bottom"
    elif loc[0] == "u":  # upper
        y += 1 - 0.035 + yoff
        va = "top"
    if loc[1] == "l":  # left
        x += 0.02 + xoff
        ha = "left"
    elif loc[1] == "r":  # right
        x += 1 - 0.02 + xoff
        ha = "right"
    version = f"v{cfg.version}"
    msg = f"MoonPIES {version}"
    xy = (x, y)
    kwargs = {"ha": ha, "va": va, "xycoords": "axes fraction", **kwargs}
    bb = {"boxstyle": "round", "fc": "w", **bbkw}
    ax.annotate(msg, xy, bbox=bb, **kwargs)
    return version
