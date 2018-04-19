from common import Segment
from roger_allocator import get_best_received_segment

def test_get_best_received_segment():
    received_times = [[10, 11, 12, 13]]
    segments = [
        [Segment(10, 0, 30, 10)],
        [Segment(11, 1, 40, 15)],
        [Segment(12, 2, 50, 20)],
        [Segment(13, 3, 60, 35)]
    ]
    
    assert (get_best_received_segment(
        received_times, segments, 0, 5) == None)
    assert (get_best_received_segment(
        received_times, segments, 0, 10) == segments[0][0])
    assert (get_best_received_segment(
        received_times, segments, 0, 12) == segments[0][2])

