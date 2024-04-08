import torch


def inflate_params_from_2d_vae(
    vae_3d_params: dict[str, torch.Tensor],
    vae_2d_params: dict[str, torch.Tensor],
    image_mode: bool = False,
) -> dict[str, torch.Tensor]:
    inflated_params: dict[str, torch.Tensor] = {}

    for key_3d in vae_3d_params.keys():
        if key_3d.startswith("loss"):
            continue

        key_2d = key_3d

        if ".down_blocks." in key_2d:
            key_2d = key_2d.replace(".down_blocks.", ".down.")
            key_2d = key_2d.replace(".convs.", ".block.")
            key_2d = key_2d.replace(".shortcut.", ".nin_shortcut.")
        elif ".mid_block." in key_2d:
            key_2d = key_2d.replace(".mid_block.", ".mid.")
            key_2d = key_2d.replace(".convs.0.", ".block_1.")
            key_2d = key_2d.replace(".convs.1.", ".block_2.")

            key_2d = key_2d.replace(".attentions.0.", ".attn_1.")
            key_2d = key_2d.replace(".group_norm.", ".norm.")
            key_2d = key_2d.replace(".to_q.", ".q.")
            key_2d = key_2d.replace(".to_k.", ".k.")
            key_2d = key_2d.replace(".to_v.", ".v.")
            key_2d = key_2d.replace(".to_out.", ".proj_out.")
        elif ".up_blocks." in key_2d:
            key_2d = key_2d.replace(".up_blocks.", ".up.")
            key_2d = key_2d.replace(".convs.", ".block.")
            key_2d = key_2d.replace(".shortcut.", ".nin_shortcut.")

            part_1, part_2 = key_2d.split(".up.")
            part_2, part_3 = part_2.split(".", maxsplit=1)
            key_2d = f"{part_1}.up.{3 - int(part_2)}.{part_3}"

        key_2d = key_2d.replace(".conv_norm_out.", ".norm_out.")
        key_2d = key_2d.replace(".downsampler.", ".downsample.")
        key_2d = key_2d.replace(".upsampler.", ".upsample.")

        assert key_2d in vae_2d_params, f"Key {key_2d} (from {key_3d}) not found in 2D VAE"

        w_3d = vae_3d_params[key_3d]
        w_2d = vae_2d_params[key_2d]
        shape_3d = w_3d.shape
        shape_2d = w_2d.shape

        if "bias" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            inflated_params[key_3d] = w_2d
        elif "norm" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            inflated_params[key_3d] = w_2d
        elif "conv" in key_2d or "nin_shortcut" in key_2d:
            if image_mode:
                new_w = w_2d
            else:
                new_w = torch.zeros(shape_3d, dtype=w_2d.dtype)
                new_w[:, :, -1, :, :] = w_2d
            inflated_params[key_3d] = new_w
        elif "attn_1" in key_2d:
            inflated_params[key_3d] = w_2d[..., 0, 0]
        else:
            raise NotImplementedError(f"Key {key_2d} (from {key_3d}) not implemented")

    return inflated_params
