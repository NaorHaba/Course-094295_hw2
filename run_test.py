import os

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from tqdm import tqdm

torch.manual_seed(0)


if __name__ == '__main__':

    # Paths to test directory
    test_dir = os.path.join("data", "test")

    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    # Create a pytorch dataset from a directory of images
    test_dataset = datasets.ImageFolder(test_dir, data_transforms)

    # Dataloader initialization
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_size = len(test_dataset)

    NUM_CLASSES = 10

    def test_model(model, dataloader, dataset_size):
        """Responsible for running the training and testidation phases for the requested model."""

        model.etest()   # Set model to etestuate mode

        running_corrects = 0

        # Iterate over data
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        acc = running_corrects.double() / dataset_size

        return acc

    # Use a prebuilt pytorch's ResNet50 model
    model_ft = models.resnet50(pretrained=False)

    # Fit the last layer for our specific task
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    model_ft = model_ft.to(device)

    # load best model weights
    best_model_wts = ''
    model_ft.load_state_dict(torch.load(best_model_wts))

    # test the model
    acc = test_model(model_ft, test_dataloader, dataset_size)

    print("Accuracy: {:.2f}%".format(acc * 100))
