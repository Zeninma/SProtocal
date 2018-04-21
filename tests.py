from common import Segment
from common import Quality
from roger_allocator import get_best_received_segment
from roger_allocator import average_quals
from roger_allocator import Allocator
import math


received_times = [[10, 100], [11, 110], [12, 120], [13, 130]]
segments = [
    [
        Segment(0, 0, 30, Quality([.1], [.01])),
        Segment(1, 0, 130, Quality([.11], [.011]))
    ],
    [
        Segment(0, 1, 40, Quality([.15], [.015])),
        Segment(1, 1, 140, Quality([.151], [.0151]))
    ],
    [
        Segment(0, 2, 50, Quality([.2], [.02])),
        Segment(1, 2, 150, Quality([.21], [.021]))
    ],
    [
        Segment(0, 3, 60, Quality([.35], [.035])),
        Segment(1, 3, 160, Quality([.351], [.0351]))
    ]
]


def test_get_best_received_segment():    
    assert (get_best_received_segment(
        received_times, segments, 0, 5) == None)
    assert (get_best_received_segment(
        received_times, segments, 0, 10) == segments[0][0])
    assert (get_best_received_segment(
        received_times, segments, 0, 12) == segments[2][0])

    
def test_average_quals():
    returned_quality = average_quals(received_times, segments, 30)
    assert math.isclose(returned_quality[0], segments[3][0].quality.psnr / 2)
    assert math.isclose(returned_quality[1], segments[3][0].quality.ssim / 2)

    returned_quality = average_quals(received_times, segments, 125)
    segment_qualities = [segments[3][0].quality, segments[2][1].quality]
    assert math.isclose(returned_quality[0], (
        (segment_qualities[0].psnr + segment_qualities[1].psnr) / 2))
    assert math.isclose(returned_quality[1], (
        (segment_qualities[0].ssim + segment_qualities[1].ssim) / 2))


def test_allocator():
    alphas = [1.31, 1.21, 1.1, 1]
    betas = [18.522, 8.82, 4.2, 2]
    allocator = Allocator(alphas, betas, segments, 1, 79)
    allocator.run_simulation()
    quals = average_quals(allocator.received_times, segments, 100)
    assert math.isclose(quals[0], 0.3505)
    assert math.isclose(quals[1], 0.03505)

test_allocator()

