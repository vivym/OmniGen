import torch
import torch.distributed as dist


def _gather(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor | None:
    tensor_list = [
        torch.empty_like(tensor) if i != dst else tensor
        for i in range(dist.get_world_size())
    ]

    rank = dist.get_rank()

    dist.gather(
        tensor,
        tensor_list if rank == dst else None,
        dst=dst,
    )

    return torch.stack(tensor_list) if rank == dst else None


def gather(
    tensors: torch.Tensor | list[torch.Tensor] | dict[torch.Tensor], dst: int = 0
) -> torch.Tensor | list[torch.Tensor] | dict[torch.Tensor]:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    if isinstance(tensors, list):
        for tensor in tensors:
            assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"

        return [_gather(tensor, dst=dst) for tensor in tensors]
    elif isinstance(tensors, dict):
        for key, tensor in tensors.items():
            assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)} (key: `{key}`)"

        return {key: _gather(tensor, dst=dst) for key, tensor in tensors.items()}
    else:
        raise ValueError(f"Unsupported type: {type(tensors)}")


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
