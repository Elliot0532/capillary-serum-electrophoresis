import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import resnet34
from PatientDataset import PatientDataset

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validate_dataset = PatientDataset(root_dir="images/test_")
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=1)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet34(num_classes=9, extra_input_dim=8).to(device)

    # load model weights
    weights_path = ("./ResNet34_multimodal.pth")
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        for data in validate_loader:
            images, numeric, labels = data
            output, attn_map = model(images.to(device), numeric.to(device))
            predict = torch.softmax(torch.squeeze(output).cpu(), dim=0)
            predict_cla = torch.argmax(predict).numpy()

    print_res = "Class: {}   Confidence: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # plt.title(print_res)
    for i in range(len(predict)):
        print("Class: {:10}   Confidence: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    image = (1 - images.cpu().numpy().squeeze()) * 255.0

    H, W = image.shape
    attn_resized = F.interpolate(attn_map, size=(H, W), mode='bicubic', align_corners=False)
    attn_resized = attn_resized[0, 0].cpu().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(8, 3))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(print_res)
    axs[0].axis('off')

    axs[1].imshow(image, cmap='gray')
    attn_im = axs[1].imshow(attn_resized, cmap='jet', alpha=0.5)
    axs[1].set_title("Image with Attention Overlay")
    axs[1].axis('off')
    # plt.colorbar(attn_im, ax=axs[1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
