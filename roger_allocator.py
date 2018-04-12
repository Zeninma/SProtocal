from heapq import heappush, heappop
import pickle
from get_quality import Segment
from get_quality import Quality


DELAY_WINDOW = 4
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

    def value(self, block_time, layer):
        size = self.segments[layer][block_time].size
        elapsed_time = self.current_time - block_time
        weight = (self.betas[layer] if elapsed_time < self.delay_window else
                  self.alphas[layer])
        return elapsed_time * (1/size) * weight

    def run_simulation(self):
        buffer = set()
        remainder_bandwidth = 0
        overflow_segment = None
        while (self.current_time < len(self.segments[0]) or
               overflow_segment is not None or
               len(buffer) > 0):
            for layer in self.segments:
                if self.current_time < len(layer):
                    buffer.add(layer[self.current_time])

            heap = []
            for segment in buffer:
                heappush(heap,
                         (-self.value(segment.time, segment.layer),
                          segment))

            transmitted = 0
            if remainder_bandwidth > self.bandwidth:
                remainder_bandwidth -= self.bandwidth
                
                # Prevent the while loop from running
                transmitted = self.bandwidth
            elif remainder_bandwidth > 0:
                transmitted = remainder_bandwidth
                remainder_bandwidth = 0
                
                layer = overflow_segment.layer
                time = overflow_segment.time
                self.received_times[layer][time] = self.current_time
                overflow_segment = None

            while transmitted < self.bandwidth and len(heap) > 0:
                to_send = heappop(heap)[1]
                buffer.remove(to_send)

                bits_to_send = to_send.size * 8
                new_size = transmitted + bits_to_send

                if new_size > self.bandwidth:
                    leftover_bandwidth = self.bandwidth - transmitted
                    remainder_bandwidth = bits_to_send - leftover_bandwidth
                    overflow_segment = to_send
                    transmitted = self.bandwidth
                    
                else:
                    transmitted += new_size

                    layer = to_send.layer
                    time = to_send.time
                    self.received_times[layer][time] = self.current_time
            
            self.current_time += 1

            
def main():
    alphas = [80, 40, 20, 10]
    betas = [8, 4, 2, 1]
    segments = pickle.load(open('segments.p', 'rb'))
    allocator = Allocator(alphas, betas, segments, DELAY_WINDOW, BANDWIDTH)
    allocator.run_simulation()
    print(allocator.received_times[0][0])


if __name__ == '__main__':
    main()
