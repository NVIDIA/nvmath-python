try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device

from nvmath._utils import get_nvrtc_version


def is_blackwell():
    cc = Device().compute_capability
    return cc.major == 12


def is_ctk12():
    try:
        major, minor, build = get_nvrtc_version()
        return major == 12
    except (ImportError, RuntimeError):
        return False
