import torch

from omni_gen.models.video_vae.autoencoder_kl import AutoencoderKL


def main():
    vae = AutoencoderKL(
        down_block_types=(
            "SpatialDownBlock3D",
            "SpatialTemporalDownBlock3D",
            "SpatialTemporalDownBlock3D",
            "SpatialTemporalDownBlock3D",
        ),
        up_block_types=(
            "SpatialUpBlock3D",
            "SpatialTemporalUpBlock3D",
            "SpatialTemporalUpBlock3D",
            "SpatialTemporalUpBlock3D",
        ),
        block_out_channels=(128, 256, 512, 512),
        mid_block_use_attention=True,
        layers_per_block=2,
        latent_channels=4,
        with_loss=False,
    )

    vae_2d_ckpt = torch.load("./vae-ft-mse-840000-ema-pruned.ckpt", map_location="cpu")

    vae_2d_keys = list(vae_2d_ckpt["state_dict"].keys())
    vae_3d_keys = list(vae.state_dict().keys())

    new_state_dict = {}

    total = 0
    for key_3d in vae_3d_keys:
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

        assert key_2d in vae_2d_keys, f"Key {key_2d} ({key_3d}) not found in 2D VAE"

        shape_3d = vae.state_dict()[key_3d].shape
        shape_2d = vae_2d_ckpt["state_dict"][key_2d].shape

        if "bias" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            new_state_dict[key_3d] = vae_2d_ckpt["state_dict"][key_2d]
        elif "norm" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            new_state_dict[key_3d] = vae_2d_ckpt["state_dict"][key_2d]
        elif "conv" in key_2d or "nin_shortcut" in key_2d:
            if shape_3d[:2] != shape_2d[:2]:
                print(key_2d, shape_3d, shape_2d)
            w = vae_2d_ckpt["state_dict"][key_2d]
            new_w = torch.zeros(shape_3d, dtype=w.dtype)
            new_w[:, :, -1, :, :] = w
            new_state_dict[key_3d] = new_w
        elif "attn_1" in key_2d:
            new_state_dict[key_3d] = vae_2d_ckpt["state_dict"][key_2d][..., 0, 0]
        else:
            raise NotImplementedError(f"Key {key_3d} ({key_2d}) not implemented")

    torch.save(new_state_dict, "./video_vae.ckpt")


if __name__ == "__main__":
    main()
