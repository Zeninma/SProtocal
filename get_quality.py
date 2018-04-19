# THIS FILE REQUIRES PYTHON 3!
# It will not work correctly with Python 2
from common import avg
from common import Segment
from common import Quality
import os
import csv
import math
import re
import pickle

FILENAME = 'I_BBB_1080p_3L{}.txt'
SEGMENT = 'BBB-I-1080p.seg{}-L{}.svc'

PSNR = 4
SSIM = 9

FPS = 24
SEG_SECONDS = 2
FRAMES_IN_SEG = FPS * SEG_SECONDS

NUM_LAYERS = 4
IGNORE_LINE_NAMES = {"Global:", "Min:"}


def get_qualities(layer):
    qualities = []

    with open(os.path.join(
            'data', 'quality', FILENAME.format(layer))) as f:
        psnrs = []
        ssims = []

        next(f)
        next(f)
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[0] == "Avg:":
                if not math.isclose(float(row[PSNR]), avg(psnrs),
                                    abs_tol=1e-05):
                    raise Exception("Wrong PSNR average for layer {}".format(
                        layer))
                if not math.isclose(float(row[SSIM]), avg(ssims),
                                    abs_tol=1e-05):
                    raise Exception("Wrong SSIM average for layer {}".format(
                        layer))
            elif row[0] not in IGNORE_LINE_NAMES:
                psnrs.append(float(row[PSNR]))
                ssims.append(float(row[SSIM]))

    for i in range(0, len(psnrs), FRAMES_IN_SEG):
        psnr_subset = psnrs[i : i+FRAMES_IN_SEG]
        ssim_subset = ssims[i : i+FRAMES_IN_SEG]

        qualities.append(Quality(psnr_subset, ssim_subset))

    return qualities

        
def main():    
    segments = []
    path = os.path.join('data', 'segs')
    for layer in range(NUM_LAYERS):
        qualities = get_qualities(layer)
        current_segments = []
        for time, quality in enumerate(qualities):
            filename = os.path.join(path, SEGMENT.format(time, layer))

            # Size in Bytes
            size = os.stat(filename).st_size
            current_segments.append(Segment(
                layer=layer, size=size, quality=quality, time=time))
        segments.append(current_segments)
    pickle.dump(segments, open('segments.p', 'wb'))

if __name__ == "__main__":
    main()
