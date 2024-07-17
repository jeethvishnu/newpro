import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])



# Define custom dataset class
class UTKFaceDataset(Dataset):
    race_mapping = {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Others"

    }
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = self.load_dataset() #lst

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def load_dataset(self):
        images = [] #paths
        labels = []

        for part_dir in sorted(os.listdir(self.root_dir)):
            if not os.path.isdir(os.path.join(self.root_dir, part_dir)):
                continue
            for filename in sorted(os.listdir(os.path.join(self.root_dir, part_dir))):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    images.append(os.path.join(part_dir, filename))
                    try:
                        # Extract race label from filename based on format [age][gender][race]_[date&time].jpg
                        race_label = int(filename.split('_')[2])
                        labels.append(race_label)
                    except (IndexError, ValueError) as e:
                        print(f"Skipping file {filename} due to error: {e}")
                        images.pop()  # Remove image entry if label extraction fails
                        continue

        if len(images) != len(labels):
            print(f"Error: Mismatch between number of images ({len(images)}) and labels ({len(labels)})")

        return images, labels
