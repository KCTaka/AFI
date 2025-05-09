import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


"""
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )

"""

class LPIPSLoss(torch.nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        
        # Load the VGG16 model
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        train_nodes, eval_nodes = get_graph_node_names(self.model)
        print("Train nodes:", train_nodes)
        print("Eval nodes:", eval_nodes)
        
        return_nodes = {
            '3': 'block1',
            '8': 'block2',
            '15': 'block3',
            '22': 'block4',
            '29': 'block5',
        }
        
        self.feature_extractor = create_feature_extractor(self.model, return_nodes=return_nodes)
        # print("Feature extractor nodes:", self.feature_extractor)
        
        # dry run
        x = torch.randn(1, 3, 224, 224)
        features = self.feature_extractor(x)
        map_feature_size = {k: v.size() for k, v in features.items()}
        print("Feature map sizes:", map_feature_size)
        
        # Imagnet normalization for (0-1)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

class LPIPS(torch.nn.Module):
    def __init__(self, net_type='vgg', reduction='mean', normalize=False, **kwargs):
        """
        Wrapper for torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity.

        Args:
            net_type (str): The network type to use. Can be one of 'alex', 'vgg', 'squeeze'.
            reduction (str): The reduction method to apply to the output. Can be one of 'mean', 'sum'.
            normalize (bool): Whether to normalize the input images to the range [-1, 1].
            **kwargs: Additional keyword arguments for LearnedPerceptualImagePatchSimilarity.
        """
        super(LPIPS, self).__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, reduction=reduction, normalize=normalize, **kwargs)
        
    def forward(self, img1, img2):
        """
        Forward pass for the LPIPS metric.

        Args:
            img1 (torch.Tensor): The first input image tensor.
            img2 (torch.Tensor): The second input image tensor.

        Returns:
            torch.Tensor: The LPIPS score between the two images.
        """
        self.lpips.update(img1, img2)
        loss = self.lpips.compute()
        self.lpips.reset()
        return loss
        

if __name__ == "__main__":
    # Example usage
    # Ensure images are in the range [-1, 1] if normalize=False (default for torchmetrics LPIPS)
    # or in [0, 1] if normalize=True (and then the class handles it)
    
    # Example with normalize=True (input images in [0,1])
    lpips_metric_normalized = LPIPS(net_type='vgg', normalize=True)
    img1_normalized = torch.rand(10, 3, 100, 100)  # N, C, H, W, values in [0, 1]
    img2_normalized = torch.rand(10, 3, 100, 100)  # N, C, H, W, values in [0, 1]
    score_normalized = lpips_metric_normalized(img1_normalized, img2_normalized)
    print(f"LPIPS score (VGG, normalized input): {score_normalized.item()}")

    # Example with normalize=False (input images in [-1,1])
    lpips_metric_manual_norm = LPIPS(net_type='alex', normalize=False)
    img1_manual_norm = (torch.rand(5, 3, 64, 64) * 2) - 1 # N, C, H, W, values in [-1, 1]
    img2_manual_norm = (torch.rand(5, 3, 64, 64) * 2) - 1 # N, C, H, W, values in [-1, 1]
    score_manual_norm = lpips_metric_manual_norm(img1_manual_norm, img2_manual_norm)
    print(f"LPIPS score (AlexNet, manually normalized input): {score_manual_norm.item()}")

    # To use on a specific device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    lpips_metric_device = LPIPS(net_type='squeeze').to(device)
    img1_device = ((torch.rand(2, 3, 128, 128) * 2) - 1).to(device)
    img2_device = ((torch.rand(2, 3, 128, 128) * 2) - 1).to(device)
    score_device = lpips_metric_device(img1_device, img2_device)
    print(f"LPIPS score (SqueezeNet, on {device}): {score_device.item()}")