import os
import pathlib
import string

import polars as pl
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Mapping for characters: digits then lowercase alphabets
char_to_idx = {ch: i for i, ch in enumerate(string.digits + string.ascii_lowercase)}
idx_to_char = {i: ch for i, ch in enumerate(string.digits + string.ascii_lowercase)}

current_dir = pathlib.Path(__file__).parent.resolve()


def get_predicted_character(output):
    # Convert model output logits to the most likely character prediction
    idx = output.argmax(dim=1).item()
    return idx_to_char[idx]


class CharacterCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CharacterCNN, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # output: (16, 160, 60)
        )
        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # output: (32, 80, 30)
        )
        # Convolutional Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # output: (64, 40, 15)
        )
        # Convolutional Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # output: (128, 20, 7)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 20 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # (16, 160, 60)
        x = self.conv2(x)  # (32, 80, 30)
        x = self.conv3(x)  # (64, 40, 15)
        x = self.conv4(x)  # (128, 20, 7)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_folder, transform=None):
        self.data_frame = pl.read_csv(csv_file).filter(pl.col("target").is_not_null())
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.row(idx, named=True)
        img_path = os.path.join(self.img_folder, row["file"])
        image = Image.open(img_path).convert("L")  # force single channel
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        target = char_to_idx[row["target"]]
        return image, target


def train_model(model, device, optimizer, criterion, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 30 == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")


if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model = CharacterCNN().to(device)

    # Setup dataset and DataLoader using the folder "x" and csv file "y"
    transform = transforms.Compose(
        [
            transforms.Resize((320, 120)),
            transforms.ToTensor(),
        ]
    )
    dataset = CaptchaDataset(
        csv_file="data/captcha_images/targets.csv", img_folder="data/captcha_images/cropped/", transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=30, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model, device, optimizer, criterion, train_loader, epochs=10)
    model_path = os.path.join(current_dir, "captcha_model.pth")
    torch.save(model.state_dict(), model_path)

    model.eval()
    # Testing with a sample inputs
    # for _ in range(5):
    #     img_no = torch.randint(0, 100, (1,)).item()
    #     crop_no = torch.randint(0, 6, (1,)).item()
    #     img_path = f"data/captcha_images/cropped/none.timesitalic.exp{img_no}.crop{crop_no}.png"
    #     image = Image.open(img_path).convert("L")
    #     image.show()
    #     image_tensor = transform(image).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         output_img = model(image_tensor)
    #     predicted_char_img = get_predicted_character(output_img)
    #     print("Predicted character:", predicted_char_img)

    # with torch.no_grad():
    #     output = model(sample_input)
    # predicted_char = get_predicted_character(output)
