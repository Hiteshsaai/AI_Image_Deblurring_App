import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import args
import kwargs
from PIL import Image
from pathlib import Path
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
# model = TheModelClass(*args, **kwargs)
# model = torch.load('outputs/model.pth')
# model.eval()

# model = TheModelClass(*args, **kwargs)

class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# trans = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(32),
#     transforms.CenterCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
#     ])

trans = transforms.ToTensor()

model = DeblurCNN()
# model.load_state_dict("./outputs/model.pth'")
model.load_state_dict(torch.load('../outputs/model.pth', map_location='cpu'))

image = Image.open(Path('../input_images/gaussian_blurred/0_IPHONE-SE_S.JPG'))

input = trans(image)

input = input.unsqueeze(0)

predict = model(input)

print(predict)

tensor1 = torch.tensor(predict,requires_grad=True)

tensor1 = tensor1.detach().numpy()

a = np.expand_dims(tensor1, axis=0)  # or axis=1
plt.imshow(a)
plt.show()
# plt.imshow(tensor1) 