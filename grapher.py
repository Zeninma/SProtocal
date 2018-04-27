import math
import numpy as np
import pdb
from common import Segment
from common import Quality
import pickle
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.WARN)

import roger_allocator


DELAY_WINDOW = 6
BANDWIDTH = 2 * 10E5


class Grapher:
    def __init__(self, alphas, beta, generator, segments):
        self.num_layers = len(segments)
        self.alphas = alphas
        self.betas = [generator(i, beta) for i in range(self.num_layers)]
        self.generator = generator
        self.segments = segments

    def get_results(self, connection_time):
        psnr = []
        ssim = []
        for alpha in self.alphas:
            current_alphas = [self.generator(i, alpha) for i in range(self.num_layers)]
            allocator = roger_allocator.Allocator(
                current_alphas, self.betas, self.segments, DELAY_WINDOW, BANDWIDTH)
            allocator.run_simulation()
            averages = roger_allocator.average_quals(
                allocator.received_times, self.segments, connection_time)
            psnr.append(averages[0])
            ssim.append(averages[1])

        return (psnr, ssim)

def plot(beta_val, alphas, psnrs, ssims, save = True):
    fname= "beta_{b_val}.png".format(b_val= str(beta_val))
    psnr_fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title("beta = {b_val}".format(b_val = beta_val))
    plt.ylabel('psnr')
    plt.plot(alphas, psnrs,'o-')
    plt.subplot(2,1,2)
    plt.xlabel('alpha')
    plt.ylabel('ssim')
    plt.plot(alphas, ssims,'o-')
    plt.savefig(fname)
    plt.cla()
    plt.close()

def main():
    alphas = np.arange(1.1, 10, 1)
    betas = [b for b in range(5)]
<<<<<<< HEAD
    def exponent(num, value):
        return value ** (num + 1)
    def multiply(num, value):
        return value * (num + 1)
    segments = pickle.load(open('large_variation_segments.p', 'rb'))

    for beta in betas:
        grapher = Grapher(alphas, beta, multiply, segments, CONNECTION_TIME)
        result = grapher.get_results()
=======

    def exponent(num, value):
        return value ** (num + 1)
    def multiplication(num, value):
        return value * (num + 1)

    segments = pickle.load(open('large_variation_segments.p', 'rb'))

    for beta in betas:
        grapher = Grapher(alphas, beta, multiplication, segments)
        result = grapher.get_results(2000)
>>>>>>> 3ea295cf7efa8562a7ae00da77e0642bae226f10
        plot(beta ,alphas, result[0], result[1])

if __name__ == "__main__":
    main()
