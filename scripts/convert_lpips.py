import torch
from safetensors.torch import save_file


def main():
    state_dict = torch.load("./tmp/vgg_lpips.pth", map_location="cpu")

    new_state_dict = {
        "0.weight": state_dict["lin0.model.1.weight"],
        "1.weight": state_dict["lin1.model.1.weight"],
        "2.weight": state_dict["lin2.model.1.weight"],
        "3.weight": state_dict["lin3.model.1.weight"],
        "4.weight": state_dict["lin4.model.1.weight"],
    }
    save_file(new_state_dict, "./tmp/lpips/vgg_lpips_linear.safetensors")


if __name__ == "__main__":
    main()
