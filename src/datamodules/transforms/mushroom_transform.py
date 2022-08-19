import torch
from src.models.cyclegan_model import CycleGanModel

class MushroomTransform(object):
    def __init__(self, model_path, direction = 'BA'):
        assert isinstance(model_path, str)
        assert direction == 'BA' or direction == 'AB'
        cg_model = CycleGanModel.load_from_checkpoint(model_path)
        cg_model.eval()
        self.model = cg_model.netG_B if direction == 'BA' else cg_model.netG_A
        self.model.eval()

    def __call__(self, image):
        with torch.no_grad():
            image = torch.unsqueeze(image, dim=0)
            image = self.model(image)
            return torch.squeeze(image.detach(),dim=0)
