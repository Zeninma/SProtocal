from collections import namedtuple


def avg(nums):
    return sum(nums) / len(nums)


Segment = namedtuple("Segment", ["time", "layer", "size", "quality"])


class Quality:
    def __init__(self, psnrs, ssims):
        self.psnr = avg(psnrs)
        self.ssim = avg(ssims)
