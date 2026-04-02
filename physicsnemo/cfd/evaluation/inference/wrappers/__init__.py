"""Model wrapper implementations; registration happens on import."""

from physicsnemo.cfd.evaluation.inference.model_registry import register_model
from physicsnemo.cfd.evaluation.inference.wrappers.domino import DominoWrapper
from physicsnemo.cfd.evaluation.inference.wrappers.fignet import FIGNetWrapper
from physicsnemo.cfd.evaluation.inference.wrappers.geotransolver import GeoTransolverWrapper
from physicsnemo.cfd.evaluation.inference.wrappers.surface_baseline import SurfaceBaselineWrapper
from physicsnemo.cfd.evaluation.inference.wrappers.transolver import TransolverWrapper
from physicsnemo.cfd.evaluation.inference.wrappers.volume_baseline import VolumeBaselineWrapper
from physicsnemo.cfd.evaluation.inference.wrappers.xmgn import XMGNWrapper

register_model("fignet", FIGNetWrapper)
register_model("xmgn", XMGNWrapper)
register_model("geotransolver", GeoTransolverWrapper)
register_model("transolver", TransolverWrapper)
register_model("domino", DominoWrapper)
register_model("surface_baseline", SurfaceBaselineWrapper)
register_model("volume_baseline", VolumeBaselineWrapper)

__all__ = [
    "FIGNetWrapper",
    "XMGNWrapper",
    "GeoTransolverWrapper",
    "TransolverWrapper",
    "DominoWrapper",
    "SurfaceBaselineWrapper",
    "VolumeBaselineWrapper",
]
