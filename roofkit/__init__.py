"""roofkit — rooftop detection & attribute extraction from Vienna open geodata.

Fuses three open sources, each blind to what the others see:
  * FMZK building footprints (vector prior),
  * ALS DSM/DGM -> nDSM (height -> slope/type/orientation, via RANSAC roof planes),
  * orthophoto (appearance -> material/PV/green/condition, via CLIP zero-shot).

Every attribute ships with a confidence score. See `roofkit.pipeline.run`.
"""
from .config import LOCATIONS, Config

__version__ = "0.1.0"
__all__ = ["Config", "LOCATIONS", "run"]


def run(config: "Config | None" = None):
    """Convenience entry point; imports the heavy pipeline lazily so `import roofkit` stays cheap."""
    from .pipeline import run as _run
    return _run(config or Config())
