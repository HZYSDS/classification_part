import torch
import clip
import torch.nn as nn
from PIL import Image
import argparse


class CLIPClassifier:
    def __init__(self, model_path, model_name="ViT-B/32", num_classes=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load clip
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)

        # classify head
        self.classifier_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        ).to(self.device)

        # load weight
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier_head.load_state_dict(checkpoint['classifier_head_state_dict'])

        self.model.eval()
        self.classifier_head.eval()

    def predict(self, image_path, class_names):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor).float()
            outputs = self.classifier_head(image_features)
            pred_class = torch.argmax(outputs, dim=1).item()

            return class_names[pred_class]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--class_names', type=str, nargs='+', required=True)

    args = parser.parse_args()

    classifier = CLIPClassifier(args.model_path)
    predicted_class = classifier.predict(args.image_path, args.class_names)
    print(predicted_class)


if __name__ == '__main__':
    main()