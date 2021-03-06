from heapq import heappush, heappop
from common import Segment
from common import Quality
import pickle
import logging

logging.basicConfig(level=logging.WARN)


DELAY_WINDOW = 4
BITS_IN_BYTE = 8
BANDWIDTH = 2 * 10E5 # One Megabit


class Allocator:
    def __init__(self, alphas, betas, segments, delay_window, bandwidth):
        self.alphas = alphas
        self.betas = betas
        self.segments = segments
        self.delay_window = delay_window
        self.current_time = 0
        self.bandwidth = bandwidth

        # indexed as [layer][timestamp]
        self.received_times = [[None for segment in layer]
                               for layer in segments]

        self.buffer = set()
        self.remainder_bandwidth = 0
        self.overflow_segment = None
        self.heap = None
        self.transmitted_in_timestep = 0

    def value(self, block_time, layer):
        if block_time == self.current_time - self.delay_window and layer == 0:
            return float('inf')
        
        size = self.segments[layer][block_time].size
        elapsed_time = self.current_time - block_time
        weight = (self.betas[layer] if elapsed_time < self.delay_window else
                  self.alphas[layer])

        # We add 1 to elapsed_time to handle the most current segment
        return (elapsed_time+1) * (1/size) * weight

    def write_time(self, segment):
        layer = segment.layer
        time = segment.time
        self.received_times[layer][time] = self.current_time

    def write_segments_for_current_time(self):
        for layer in self.segments:
            if self.current_time < len(layer):
                self.buffer.add(layer[self.current_time])
        
    def write_buffer_to_heap(self):
        """Clear self.heap and write the contents of self.buffer to self.heap"""
        # This program could be made more efficient if we found a way to re-use part of
        # the heap. Unfortunately this is difficult because segment values change

        self.heap = []
        num_lowest_layer = 0
        for segment in self.buffer:
            # We use negative value to get a max-first heap
            heappush(self.heap,
                     (-self.value(segment.time, segment.layer),
                      segment))
            if segment.layer == 0:
                num_lowest_layer += 1

    def send_leftover_segment(self):
        """We want to send a leftover segment from the last timestep if there was one.
        
        If there is a leftover segment and enough bandwidth to send all of it, we will.
        If there is only enough bandwidth to send part of it, we will send as much as
        we can."""

        if self.remainder_bandwidth > self.bandwidth:
            logging.info("Using all bandwidth for timestep to send part of overflow "
                         "segment")
            self.remainder_bandwidth -= self.bandwidth
            self.transmitted_in_timestep = self.bandwidth
                
        elif self.remainder_bandwidth > 0:
            logging.info("Completed sending overflow segment")
            self.transmitted_in_timestep = self.remainder_bandwidth
            self.remainder_bandwidth = 0
                
            self.write_time(self.overflow_segment)
            self.overflow_segment = None

    def send_segs_from_heap(self):
        while self.transmitted_in_timestep < self.bandwidth and len(self.heap) > 0:
            to_send = heappop(self.heap)[1]
            self.buffer.remove(to_send)

            bits_to_send = to_send.size * BITS_IN_BYTE
            new_size = self.transmitted_in_timestep + bits_to_send

            if new_size > self.bandwidth:
                logging.info("Segment from layer %d and time %d too large to send at "
                             "timestep %d. Storing.", to_send.layer, to_send.time,
                             self.current_time)
                self.remainder_bandwidth = new_size - self.bandwidth
                self.overflow_segment = to_send
                self.transmitted_in_timestep = self.bandwidth
                    
            else:
                self.transmitted_in_timestep += new_size
                self.write_time(to_send)
            
    def run_simulation(self):
        while (self.current_time < len(self.segments[0]) or
               self.overflow_segment is not None or
               len(self.buffer) > 0):
            logging.info("Simulation at timestep %d", self.current_time)
            self.write_segments_for_current_time()
            self.write_buffer_to_heap()

            self.transmitted_in_timestep = 0
            self.send_leftover_segment()
            self.send_segs_from_heap()
            
            self.current_time += 1


def found_lower(found_layer, frame, timestep, received_times):
    # This is inefficient because it is run every time
    # But we should only have to run it once unless we have
    # really bad values of alpha and beta
    for layer in range(found_layer):
        if received_times[layer][frame] > timestep:
            logging.warn(
                "Received %d %d but not %d %d", found_layer,
                frame, layer, frame)
            return False
    return True

            
def get_best_received_segment(received_times, segments, frame, timestep):
    """Find the highest layer that was received at or before timestep"""
    for layer in range(len(segments)-1, -1, -1):
        if (received_times[layer][frame] is not None and
            received_times[layer][frame] <= timestep and
            found_lower(layer, frame, timestep, received_times)):
            return segments[layer][frame]
    return None

            
def average_quals(received_times, segments, join_time):
    total_psnr = 0
    total_ssim = 0

    num_frames = len(segments[0])
    for frame in range(num_frames):
        timestep = join_time + frame
        best_received_segment = get_best_received_segment(
            received_times, segments, frame, timestep)
        if best_received_segment is not None:
            total_psnr += best_received_segment.quality.psnr
            total_ssim += best_received_segment.quality.ssim

    return (total_psnr / num_frames, total_ssim / num_frames)
            
            
def main():
    alphas = [80, 40, 20, 10]
    betas = [8, 4, 2, 1]
    segments = pickle.load(open('large_variation_segments.p', 'rb'))
    allocator = Allocator(alphas, betas, segments, DELAY_WINDOW, BANDWIDTH)
    allocator.run_simulation()

    for layer in allocator.received_times:
        print(*layer)
    
    print(average_quals(allocator.received_times, segments, DELAY_WINDOW + 30))


if __name__ == '__main__':
    main()
