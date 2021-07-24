import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


classes = ['stitch']
device = 'cuda'if torch.cuda.is_available() else 'cpu'


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def model_fn(model_dir):
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (note: +1 because of the __background__ class)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
    model.to(device)

    return model


def predict_fn(input_data, model):
    model.eval()

    with torch.no_grad():
        defaults = transforms.Compose([transforms.ToTensor()])
        images = defaults(input_data)
        images = images.to(device)

        preds = model(images)
        # Send predictions to CPU if not already
        preds = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds]
    
    results = []
    for pred in preds:
        # Convert predicted ints into their corresponding string labels
        result = ([classes[val] for val in pred['labels']], pred['boxes'], pred['scores'])
        results.append(result)

    return results[0]