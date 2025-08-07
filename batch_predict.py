import os
import json
import sys
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms

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
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=9).to(device)

    # load model weights
    weights_path = "./ResNet34_singlemodal.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_numeric, val_labels = val_data
            outputs, attn_map = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('Total Test Accuracy: %.3f' % val_accurate)

if __name__ == '__main__':
    main()
