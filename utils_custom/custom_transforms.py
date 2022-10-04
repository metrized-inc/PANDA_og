import torchvision.transforms.functional as F

class AffineScale(object):
    # Apply an affine transform
    def __init__(self, scale):
        assert isinstance(scale, float)
        self.scale = scale

    def __repr__(self):
        return 'Scale ({:.2f})'.format(self.scale)

    def __call__(self, img):
        img = F.affine(img, angle=0, translate=(0, 0), scale=self.scale, shear=0)
        return img

class HueShift(object):
    def __init__(self, offset):
        self.offset = offset

    def __repr__(self):
        return 'Hue Offset ({:.2f})'.format(self.offset)

    def __call__(self, img):
        img = F.adjust_hue(img, self.offset)
        return img

class SatShift(object):
    def __init__(self, factor):
        self.factor = factor

    def __repr__(self):
        return 'Scale ({:.2f})'.format(self.factor)

    def __call__(self, img):
        img = F.adjust_saturation(img, self.factor)
        return img