import accelerate
from packaging import version


def apply_forward_hook(method):
    """
    Decorator that applies a registered CpuOffload hook to an arbitrary function rather than `forward`. This is useful
    for cases where a PyTorch module provides functions other than `forward` that should trigger a move to the
    appropriate acceleration device. This is the case for `encode` and `decode` in [`AutoencoderKL`].

    This decorator looks inside the internal `_hf_hook` property to find a registered offload hook.

    :param method: The method to decorate. This method should be a method of a PyTorch module.
    """
    accelerate_version = version.parse(accelerate.__version__).base_version
    if version.parse(accelerate_version) < version.parse("0.17.0"):
        return method

    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "pre_forward"):
            self._hf_hook.pre_forward(self)
        return method(self, *args, **kwargs)

    return wrapper
