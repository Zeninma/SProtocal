from common import Segment
from common import Quality
import pickle
import logging
logging.basicConfig(level=logging.WARN)

import roger_allocator


DELAY_WINDOW = 6
BANDWIDTH = 20 * 10E5
CONNECTION_TIME = 50


class Grapher:
    def __init__(self, alphas, beta, generator, segments, connection_time):
        self.num_layers = len(segments)
        self.alphas = alphas
        self.betas = [generator(i, beta) for i in range(self.num_layers)]
        self.generator = generator
        self.segments = segments
        self.connection_time = connection_time

    def get_results(self):
        psnr = []
        ssim = []
        for alpha in self.alphas:
            current_alphas = [self.generator(i, alpha) for i in range(self.num_layers)]
            allocator = roger_allocator.Allocator(
                current_alphas, self.betas, self.segments, DELAY_WINDOW, BANDWIDTH)
            allocator.run_simulation()
            averages = roger_allocator.average_quals(
                allocator.received_times, self.segments, self.connection_time)
            psnr.append(averages[0])
            ssim.append(averages[1])

        return (psnr, ssim)

def main():
    alphas = [1.1, 2, 10, 100]
    beta = 2
    def generator(num, value):
        return value ** (num + 1)
    segments = pickle.load(open('segments.p', 'rb'))

    grapher = Grapher(alphas, beta, generator, segments, CONNECTION_TIME)
    print(grapher.get_results())

if __name__ == "__main__":
    main()
