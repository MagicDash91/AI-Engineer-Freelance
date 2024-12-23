import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import uvicorn

# Initialize FastAPI app
app = FastAPI()

categories = [
    'n02108915-French_bulldog', 'n02108551-Tibetan_mastiff', 'n02093859-Kerry_blue_terrier',
    'n02112350-keeshond', 'n02089078-black-and-tan_coonhound', 'n02099712-Labrador_retriever',
    'n02106166-Border_collie', 'n02110627-affenpinscher', 'n02088238-basset', 'n02101556-clumber',
    'n02106550-Rottweiler', 'n02091134-whippet', 'n02111129-Leonberg', 'n02096177-cairn',
    'n02091467-Norwegian_elkhound', 'n02098413-Lhasa', 'n02101006-Gordon_setter',
    'n02113624-toy_poodle', 'n02085782-Japanese_spaniel', 'n02108089-boxer',
    'n02102177-Welsh_springer_spaniel', 'n02093647-Bedlington_terrier', 'n02096585-Boston_bull',
    'n02112018-Pomeranian', 'n02093991-Irish_terrier', 'n02110806-basenji',
    'n02100877-Irish_setter', 'n02096437-Dandie_Dinmont', 'n02108422-bull_mastiff',
    'n02096051-Airedale', 'n02086646-Blenheim_spaniel', 'n02108000-EntleBucher',
    'n02116738-African_hunting_dog', 'n02085620-Chihuahua', 'n02094258-Norwich_terrier',
    'n02099429-curly-coated_retriever', 'n02107142-Doberman', 'n02101388-Brittany_spaniel',
    'n02107312-miniature_pinscher', 'n02109961-Eskimo_dog', 'n02097047-miniature_schnauzer',
    'n02094114-Norfolk_terrier', 'n02105855-Shetland_sheepdog', 'n02110185-Siberian_husky',
    'n02097130-giant_schnauzer', 'n02087046-toy_terrier', 'n02113186-Cardigan',
    'n02115641-dingo', 'n02102040-English_springer', 'n02090622-borzoi', 'n02110063-malamute',
    'n02106382-Bouvier_des_Flandres', 'n02099849-Chesapeake_Bay_retriever', 'n02088466-bloodhound',
    'n02092339-Weimaraner', 'n02090721-Irish_wolfhound', 'n02112706-Brabancon_griffon',
    'n02099601-golden_retriever', 'n02105251-briard', 'n02115913-dhole',
    'n02095314-wire-haired_fox_terrier', 'n02093428-American_Staffordshire_terrier',
    'n02095889-Sealyham_terrier', 'n02109525-Saint_Bernard', 'n02102318-cocker_spaniel',
    'n02105412-kelpie', 'n02088364-beagle', 'n02091244-Ibizan_hound', 'n02106662-German_shepherd',
    'n02094433-Yorkshire_terrier', 'n02091635-otterhound', 'n02098286-West_Highland_white_terrier',
    'n02113023-Pembroke', 'n02100735-English_setter', 'n02113799-standard_poodle',
    'n02090379-redbone', 'n02092002-Scottish_deerhound', 'n02110958-pug', 'n02109047-Great_Dane',
    'n02097209-standard_schnauzer', 'n02107574-Greater_Swiss_Mountain_dog',
    'n02111277-Newfoundland', 'n02095570-Lakeland_terrier', 'n02086240-Shih-Tzu',
    'n02099267-flat-coated_retriever', 'n02100236-German_short-haired_pointer', 'n02107908-Appenzeller',
    'n02105162-malinois', 'n02111889-Samoyed', 'n02113978-Mexican_hairless',
    'n02113712-miniature_poodle', 'n02093256-Staffordshire_bullterrier', 'n02086910-papillon',
    'n02112137-chow', 'n02105505-komondor', 'n02105641-Old_English_sheepdog',
    'n02087394-Rhodesian_ridgeback', 'n02097474-Tibetan_terrier', 'n02100583-vizsla',
    'n02097298-Scotch_terrier', 'n02105056-groenendael', 'n02098105-soft-coated_wheaten_terrier',
    'n02097658-silky_terrier', 'n02102480-Sussex_spaniel', 'n02091032-Italian_greyhound',
    'n02088094-Afghan_hound', 'n02096294-Australian_terrier', 'n02089973-English_foxhound',
    'n02106030-collie', 'n02104365-schipperke', 'n02093754-Border_terrier', 'n02089867-Walker_hound',
    'n02085936-Maltese_dog', 'n02111500-Great_Pyrenees', 'n02104029-kuvasz', 'n02086079-Pekinese',
    'n02102973-Irish_water_spaniel', 'n02107683-Bernese_mountain_dog', 'n02091831-Saluki',
    'n02088632-bluetick'
]

class DogsCNN(nn.Module):
    def __init__(self, num_classes):
        super(DogsCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 input channels (RGB), 32 output channels
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after convolution
        self.pool1 = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 kernel

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Calculate the size of the feature map after the last pooling layer
        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, self.bn1, self.pool1,
                                   self.conv2, self.bn2, self.pool2,
                                   self.conv3, self.bn3, self.pool3,
                                   self.conv4, self.bn4, self.pool4)
        self._get_to_linear(227)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 1024)  # Reduced size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def _get_to_linear(self, size):
        x = torch.randn(1, 3, size, size)
        x = self.convs(x)
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Correct number of labels
num_classes = 20

# Initialize the model with the correct number of classes
model = DogsCNN(num_classes=num_classes)

# Load the checkpoint, skipping mismatched layers
checkpoint = torch.load("dogs_cnn_model.pth", map_location=torch.device("cpu"), weights_only=True)
model.load_state_dict(checkpoint, strict=False)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Adjust based on your model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def read_image(file) -> Image.Image:
    """Convert uploaded file to a PIL Image."""
    image = Image.open(BytesIO(file)).convert("RGB")
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict the category of an uploaded image."""
    try:
        image = read_image(await file.read())
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = categories[predicted.item()]
        return {"filename": file.filename, "predicted_label": label}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
