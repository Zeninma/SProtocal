from collections import namedtuple
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

SearchResult = namedtuple("SearchResult",["result_list","alpha","k_val", "generator" ,"bandwidth"])

SimResult = namedtuple("SimResult",["psnr","ssim","alpha","k_val","connection_time","bandwidth"])

AvgQuality = namedtuple("AvgQuality", ["psnr", "ssim","size"])

DELAY_WINDOW = 6
BANDWIDTH = 2 * 10E5


class Grapher:
    def __init__(self, alphas, beta, generator, segments):
        self.num_layers = len(segments)
        self.alphas = alphas
        self.betas = [generator(i, beta, self.num_layers) for i in range(self.num_layers)]
        self.generator = generator
        self.segments = segments
        self.layers_avg_quality = [] # average qualities for each layer
        self.slice_num = len(segments[0])
        
    def get_avg_qualities(self):
        if len(self.layers_avg_quality) == 0:
            for i in range(len(self.segments)):
                curr_psnrs = []
                curr_ssims = []
                curr_layer = self.segments[i]
                curr_sizes = []
                # curr_qualities = [seg.quality for seg in curr_layer]
                for seg in curr_layer:
                    curr_psnrs.append(seg.quality.psnr)
                    curr_ssims.append(seg.quality.ssim)
                    curr_sizes.append(seg.size)
                curr_psnr_avg = sum(curr_psnrs)/self.slice_num
                curr_ssim_avg = sum(curr_ssims)/self.slice_num
                self.layers_avg_quality.append(AvgQuality(curr_psnr_avg,curr_ssim_avg))
        else:
            return self.layers_avg_quality

    def get_results(self, connection_times, bandwidth = BANDWIDTH):
        result = [[[],[]] for i in range(len(connection_times))] # [[results for connection_time1],[results for connection_time2],...]
        for alpha in self.alphas:
            current_alphas = [self.generator(i, alpha, self.num_layers) for i in range(self.num_layers)]
            allocator = roger_allocator.Allocator(
                current_alphas, self.betas, self.segments, DELAY_WINDOW, bandwidth)
            allocator.run_simulation()
            for idx, connection_time in enumerate(connection_times):
                averages = roger_allocator.average_quals(
                    allocator.received_times, self.segments, connection_time)
                result[idx][0].append(averages[0]) # psnr
                result[idx][1].append(averages[1]) # ssim

        return result

    def search_results(self, alpha, k_val, generator, connection_times, bandwidth = BANDWIDTH):
        result = [] #[(psnr, ssim for conn_time_1), (psnr,ssim for conn_time_2)...]
        beta = alpha * k_val
        alphas = [generator(i, alpha, self.num_layers) for i in range(self.num_layers)]
        betas = [generator(i, beta, self.num_layers) for i in range(self.num_layers)]
        allocator = roger_allocator.Allocator(
                alphas, betas, self.segments, DELAY_WINDOW, bandwidth)
        allocator.run_simulation()
        for connection_time in connection_times:
            averages = roger_allocator.average_quals(
                allocator.received_times, self.segments, connection_time)
            result.append((averages[0], averages[1])) # (psnr, ssim)
        
        return result

    def search_resultsb(self, alpha, beta, generator, connection_times, bandwidth = BANDWIDTH):
        result = [] #[(psnr, ssim for conn_time_1), (psnr,ssim for conn_time_2)...]
        alphas = [generator(i, alpha, self.num_layers) for i in range(self.num_layers)]
        betas = [generator(i, beta, self.num_layers) for i in range(self.num_layers)]
        allocator = roger_allocator.Allocator(
                alphas, betas, self.segments, DELAY_WINDOW, bandwidth)
        allocator.run_simulation()
        for connection_time in connection_times:
            averages = roger_allocator.average_quals(
                allocator.received_times, self.segments, connection_time)
            result.append((averages[0], averages[1])) # (psnr, ssim)
        
        return result

def plot(beta_val, alphas, psnrs, ssims, connection_time, alg_name,bandwidth = BANDWIDTH, save = True):
    connection_time = connection_time * 2
    bandwidth = (bandwidth/2.0)
    sub_1_ymax = max(psnrs)  + 1
    fname= "beta_{b_val}_conn_{conn_time}_bw_{bw}_{alg}.png".\
        format(b_val= str(beta_val), conn_time = str(connection_time),bw = str(bandwidth), alg = alg_name)
    plt.title("beta={b_val},connection_time={conn_time}s,bw={bw},{alg}".\
        format(b_val= str(beta_val), conn_time = str(connection_time), bw = str(bandwidth),alg = alg_name))

    sub_1 = plt.subplot(2,1,1)
    sub_2 = plt.subplot(2,1,2)
    
    sub_1.plot(alphas, psnrs,'o-')
    sub_2.plot(alphas, ssims,'o-')

    sub_1.set_ylabel('psnr')
    sub_2.set_xlabel('alpha')
    sub_2.set_ylabel('ssim')
    
    sub_1.set_ylim(0.0, sub_1_ymax)
    sub_1.set_xlim(xmin = 0.0)
    sub_2.set_ylim(0.0, 1.0)
    sub_2.set_xlim(xmin = 0.0)

    plt.savefig(fname)
    plt.cla()
    plt.close()

def main():
    def exponent(num, value, num_layers):
        return value ** (num_layers - num)
    def multiplication(num, value, num_layers):
        return value * (num_layers - num)

    # alphas = np.arange(1.1, 30.0, 3.0)
    # k_vals = np.arange(2.0,100.0, )
    alphas = [1.1, 2.0, 3.0 ,4.0]
    k_vals = [2.0,3.0,4.0,5.0]
    segments = pickle.load(open('large_variation_segments.p', 'rb'))
    generators = [exponent, multiplication]
    connection_times = [DELAY_WINDOW, 200, 1000]
    grapher = Grapher(alphas, 0.0, multiplication, segments)
    search_steps = 8

    result = search(grapher ,alphas, k_vals, generators, connection_times, search_steps)
    print(result)

def search_alpha(grapher, alphas, k_val, generator,connection_times):
    max_psnr = -1.0
    result_list = [] # a list of result for different connection times
    result_alpha = 0
    for alpha in alphas:
        curr_result = grapher.search_results(alpha, k_val, generator, connection_times)
        sum_psnr = 0.0
        sum_ssim = 0.0
        for pair in curr_result:
            sum_psnr += pair[0]
            sum_ssim += pair[1]
        curr_psnr = sum_psnr/float(len(curr_result))
        if curr_psnr > max_psnr:
            result_alpha = alpha
            max_psnr = curr_psnr
            result_list = curr_result

    return result_list, result_alpha
            

def search_k_val(grapher, alpha, k_vals, generator, connection_times):
    max_psnr = -1.0
    result_list = []
    result_k_val = 0
    for k_val in k_vals:
        curr_result = grapher.search_results(alpha, k_val, generator, connection_times)
        sum_psnr = 0.0
        sum_ssim = 0.0
        for pair in curr_result:
            sum_psnr += pair[0]
            sum_ssim += pair[1]
        curr_psnr = sum_psnr/float(len(curr_result))
        if curr_psnr > max_psnr:
            result_k_val = k_val
            max_psnr = curr_psnr
            result_list = curr_result

    return result_list, result_k_val

def search_gen(grapher, alpha, k_val, generators, connection_times):
    max_psnr = -1.0
    result_list = []
    result_generator = None
    for generator in generators:
        curr_result = grapher.search_results(alpha, k_val, generator, connection_times)
        sum_psnr = 0.0
        sum_ssim = 0.0
        for pair in curr_result:
            sum_psnr += pair[0]
            sum_ssim += pair[1]
        curr_psnr = sum_psnr/float(len(curr_result))
        if curr_psnr > max_psnr:
            result_generator = generator
            max_psnr = curr_psnr
            result_list = curr_result

    return result_list, result_generator

def search(grapher ,alphas, k_vals, generators, connection_times, search_steps):
    curr_alpha = alphas[0]
    curr_k_val = k_vals[0]
    curr_generator = generators[1]
    results = []
    for i in range(search_steps):
        gen_name = ""
        _, curr_alpha = search_alpha(grapher, alphas, curr_k_val, curr_generator, connection_times)
        curr_result_list, curr_k = search_k_val(grapher, curr_alpha, k_vals, curr_generator, connection_times)
        if curr_generator == generators[0]:
            gen_name = "exp"
        else:
            gen_name = "multi"
        
        curr_result = SearchResult(curr_result_list,curr_alpha, curr_k, gen_name, BANDWIDTH)
        results.append(curr_result)

    pickle.dump(results, open('search_info.p', 'wb'))
    return results[-1]
            
if __name__ == "__main__":
    main()
