from heapq import heappush, heappop
import pickle
from get_quality import Segment
from get_quality import Quality


DELAY_WINDOW = 4
BITS_IN_BYTE = 8
BANDWIDTH = 10E5 # One Megabit


class Allocator:
    def __init__(self, alphas, betas, segments, delay_window, bandwidth):
        self.alphas = alphas
        self.betas = betas
        self.segments = segments
        self.delay_window = delay_window
        self.current_time = 0
        self.bandwidth = bandwidth
        self.received_times = [[None for segment in layer]
                               for layer in segments]
        self.buffer = set()
        self.remainder_bandwidth = 0
        self.overflow_segment = None
        self.heap = None

    def value(self, block_time, layer):
        size = self.segments[layer][block_time].size
        elapsed_time = self.current_time - block_time
        weight = (self.betas[layer] if elapsed_time < self.delay_window else
                  self.alphas[layer])
        return elapsed_time * (1/size) * weight

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
        self.heap = []
        for segment in self.buffer:
            heappush(self.heap,
                     (-self.value(segment.time, segment.layer),
                      segment))
    
    def run_simulation(self):
        while (self.current_time < len(self.segments[0]) or
               self.overflow_segment is not None or
               len(self.buffer) > 0):
            self.write_segments_for_current_time()
            self.write_buffer_to_heap()

            transmitted = 0
            if self.remainder_bandwidth > self.bandwidth:
                self.remainder_bandwidth -= self.bandwidth
                transmitted = self.bandwidth
                
            elif self.remainder_bandwidth > 0:
                transmitted = self.remainder_bandwidth
                self.remainder_bandwidth = 0
                
                self.write_time(self.overflow_segment)
                self.overflow_segment = None

            while transmitted < self.bandwidth and len(self.heap) > 0:
                to_send = heappop(self.heap)[1]
                self.buffer.remove(to_send)

                bits_to_send = to_send.size * BITS_IN_BYTE
                new_size = transmitted + bits_to_send

                if new_size > self.bandwidth:
                    leftover_bandwidth = self.bandwidth - transmitted
                    self.remainder_bandwidth = bits_to_send - leftover_bandwidth
                    self.overflow_segment = to_send
                    transmitted = self.bandwidth
                    
                else:
                    transmitted += new_size
                    self.write_time(to_send)
            
            self.current_time += 1


def get_best_received_segment(received_times, segments, frame, timestep):
    for layer in range(len(segments), 0, -1):
        if received_times[layer][frame] <= timestep:
            return segments[layer][frame]
    return None

            
def average_quals(received_times, segments, join_time):
    total_psnr = 0
    total_ssim = 0
    for frame in range(len(segments[0])):
        timestep = join_time + frame
        best_received_segment = get_best_received_segment(
            received_times, segments, frame, timestep)
        if best_received_segment is not None:
            total_psnr += best_received_segment.psnr
            total_ssim += best_received_segment.ssim
            
            
def main():
    alphas = [80, 40, 20, 10]
    betas = [8, 4, 2, 1]
    segments = pickle.load(open('segments.p', 'rb'))
    allocator = Allocator(alphas, betas, segments, DELAY_WINDOW, BANDWIDTH)
    allocator.run_simulation()
    print(allocator.received_times[0][0])


if __name__ == '__main__':
    main()
