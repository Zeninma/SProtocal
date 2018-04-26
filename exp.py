from collections import deque
import math
import pdb, traceback, sys
import matplotlib.pyplot as plt
from get_quality import Segment, Quality
import pickle

# @TODO: write a function to read a csv file of chunks. Every chunk will be converted to a Chunk object,
# and all the Chunk objects will be wrapped into a Buffer Object.
# @TODO: write a function to read a csv file of upload bandwidth trace and return a BandWidth object
SEGPATH = "segments.p"
LAYERNUM = 4
DEFAULTLEN = 2


def loadSegments(path = SEGPATH, layerNum = LAYERNUM):
    with open(path, "rb") as f:
        segments = pickle.load(f)

    streamChunks = Buffer(layerNum)
    for layer in range(LAYERNUM):
        layerSegs = segments[layer]
        for seg in layerSegs:
            newChunk = Chunk(seg.size*8.0,seg.layer,seg.quality, DEFAULTLEN)
            streamChunks.addChunk(newChunk)

    return streamChunks


# class Quality:
#     '''
#     Quality -- a wrapper object containing metrics that are used to represent the quality
#     of the video
#     '''
#     quality = {}

class BandWidth:
    '''
    BandWidth - A wrapper of array, where the ith element indicates the bandwidth at i*unitTime
    '''
    unitTime = 1.0
    def __init__(self, unit, bwList):
        '''

        :param unit: number of seconds each element represents
        '''
        self.unitTime = float(unit)
        self.bandWidthQueue = deque(bwList)

    def __getitem__(self, index):
        return self.bandWidthQueue[index]

class Chunk:
    '''
    Chunk -- represents a chunk of video with fixed length
    '''

    def __init__(self, newSize, newLayer, newQuality, newTimeLen):
        '''

        :param newSize: size of the chunk in Mb
        :param newLayer: the layer this chunk belongs to
        :param newQuality: quality of the chunk, might be an object wraps several
        metrics for quality measurement wrapped as a Quality object
        :param newTimeLen: number of second of video this chunk contains
        '''
        self.counter = 0 # indicate which round the chunk is added to the stream.streamBuffer
        self.size = newSize # the number of bytes of the content
        self.layer = newLayer # which layer does this chunk belong to
        self.quality = newQuality
        self.timeLen = newTimeLen # time length for each content

    def setSent(self, sentTime):
        # Set the time stamp when it is sent
        self.sentTime = sentTime # This is set when the packet is sent to the server

    def setCounter(self, counter):
        # Indicating which segment this chunk belongs to
        self.counter = counter

class Buffer:
    '''
    Buffer -- represents the uploader's in memory buffer,
    which holds encoded video chunks
    '''

    def __init__(self, numLayer = 3):
        self.layerNum = numLayer
        self.buffer = []
        for i in range(numLayer):
            self.buffer.append(deque())
        self.layerNum = numLayer

    def __getitem__(self, index):
        return self.buffer[index]

    def addChunk(self, chunk):
        layer = chunk.layer
        self.buffer[layer].append(chunk)

    def addStreamChunks(self, streamChunks, totalSliceNum, chunkCount):
        '''
        :param streamChunks - the buffer contains all the chunks that will be produced by video encoder
        :param timeElapse - number of seconds passed since the last addStreamChunks happened
        :return: True if the StreamChunks is empty Flase otherwise
        '''
        totalSliceNum = min(totalSliceNum, len(streamChunks[0]))

        for layer in range(streamChunks.layerNum):
            for sliceNum in range(totalSliceNum):
                newChunk = streamChunks[layer].popleft()
                newChunk.setCounter(chunkCount + sliceNum + 1)
                self.buffer[layer].append(newChunk)

        return streamChunks.empty()

    def empty(self):
        # Check if the whole buffer is totally empty
        result = True
        for layer in range(self.layerNum):
            result = result and len(self.buffer[layer]) == 0
        return result

    def remainChunks(self):
        chunkNum = 0
        for i in range(self.layerNum):
            chunkNum += len(self.buffer[i])
        return chunkNum

    def converSegToChunk(self, segments):
        '''

        :param segments: A 2d array of GQ.Segment
        :return:
        '''


class Stream:
    '''
    Stream - simulate the stream process
    '''
    def __init__(self, numLayer, latency, streamChunks, bwList, algParam):
        '''

        :param numLayer - number of layers produced by the encoder
        :param latency - latency for streaming (around 10 sec is acceptable)
        :param streamChunks - chunks for upload wrapped as a Buffer object
        :param bwList - The bandwidth for uploading
        :param algParam - a dictionary for algorithm's parameters
        '''
        self.timeCounter = 0.0
        self.numLayer = numLayer
        self.latency = latency
        self.streamChunks = streamChunks
        self.bwList = bwList
        self.streamBuffer = Buffer(self.numLayer) # representing chunks in memory buffer at self.timeCounter
        self.outputBuffer = Buffer(self.numLayer) # represent the chunks received by the server
        self.chunkLen = streamChunks[0][0].timeLen
        self.latencyWinSize = int(math.ceil(self.latency*1.0/self.chunkLen))
        self.chunkCount = 0 # Indicating which round the chunk at the head of streamBuffer is added
        self.initAlg()
        # first fill the buffer



    def initAlg(self):
        '''
        @TODO: modify to take a real algParam
        Initialize the coefficient
        :return:
        '''
        self.headsCo = [360.0,60.0,10.0,1.6]
        self.tailsCo = [9.0,3.0,1.0,0.1]


    def run(self):
        # the starting period, where self.latency seconds of video will be added to the stream buffer
        # self.streamBuffer.addStreamChunks(self.streamChunks, self.latencyWinSize, self.chunkCount)
        # self.chunkCount += self.latencyWinSize
        # Debug Purpose
        chunk_sent = 0
        prevTime = self.timeCounter
        while not self.streamChunks.empty() or not self.streamBuffer.empty():
            if not self.streamBuffer.empty():
                if chunk_sent == 114:
                    pdb.set_trace()
                currChunk = self.getNextChunk()
                self.send(currChunk)
                chunk_sent += 1
                print("chunk sent at {0: .2f} at layer {1: =5} lowest len {2: =5}".\
                      format(self.timeCounter, currChunk.layer, len(self.streamBuffer[0])))
                sliceNum = int((self.timeCounter - prevTime) / self.chunkLen)
                if sliceNum > 0:
                    self.streamBuffer.addStreamChunks(self.streamChunks, sliceNum, self.chunkCount)
                    self.chunkCount += sliceNum
                    prevTime = self.timeCounter
            else:
                # increment timer forward and add new chunks into stream buffer
                self.timeCounter = (math.floor(self.timeCounter/self.chunkLen)+1)*self.chunkLen
                self.streamBuffer.addStreamChunks(self.streamChunks, 1, self.chunkCount)
                self.chunkCount += 1
                prevTime = self.timeCounter


    def send(self, chunk):
        '''
        calculate the time needed to send the chunk, increment the time counter,
        set the arrive time for chunk and put it into self.outputBuffer
        :param chunk - the chosen chunk to be sent to the server
        :return:
        '''
        currBWIdx = int(self.timeCounter/self.bwList.unitTime)

        timeLeft = (currBWIdx+1) * self.bwList.unitTime - self.timeCounter
        currBW = self.bwList[currBWIdx]
        size = chunk.size
        while True:
            if size - currBW * timeLeft > 0: # determine whether to update current bandwidth
                size -= currBW * timeLeft
                self.timeCounter += timeLeft
                currBWIdx += 1
                currBW = self.bwList[currBWIdx]
                timeLeft = self.bwList.unitTime
            else:
                self.timeCounter += size*1.0/currBW
                chunk.setSent(self.timeCounter)
                self.outputBuffer.addChunk(chunk)
                # Add new chunk into streamBuffer
                # if stream chuncks is not empty
                return

    def getNextChunk(self):
        '''
        Use our algorithm to determine the next chunk to send
        :return: The chunk to send next
        '''
        # for layer in range(self.numLayer):
        #     exist, result = self.latencyWinFirst(layer)
        #     if exist:
        #         return result
        exist, result = self.latencyWinFirst(0)
        if exist:
            return result

        heads = []
        tails = []

        for layer in range(self.numLayer):
            heads.append(self.getHead(layer))
            tails.append(self.getTail(layer))
        maxPair = heads[0]

        for pair in heads:
            if pair[2] > maxPair[2]:
                maxPair = pair
        for pair in tails:
            if pair[2] > maxPair[2]:
                maxPair = pair
        try:
            resultChunk = self.streamBuffer[maxPair[0]][maxPair[1]]
        except:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)

        del self.streamBuffer[maxPair[0]][maxPair[1]]
        assert resultChunk is not None # Sanity Check

        return resultChunk


    def getHead(self, layerNum):
        currLayer = self.streamBuffer[layerNum]
        if len(currLayer) == 0:
            return (layerNum, 0, -1)
        idx = len(currLayer) - 1
        while idx > 0 and currLayer[idx].counter > self.chunkCount - self.latencyWinSize + 1:
            idx -= 1
        # val = (self.chunkCount - currLayer[0].counter)*self.chunkLen*self.headsCo[layerNum]/currLayer[idx].size
        val = (self.chunkCount - currLayer[0].counter) * self.chunkLen * self.headsCo[layerNum]
        return (layerNum, idx, val)

    def getTail(self,layerNum):
        currLayer = self.streamBuffer[layerNum]
        if len(currLayer) == 0:
            return (layerNum, 0, -1)
            # @TODO: Here use negative num to indicate the current layer is empty, might need
            # Might need better way to indicate this when the value can be negative
        # val = (self.chunkCount - currLayer[0].counter)*self.chunkLen * self.tailsCo[layerNum]/currLayer[0].size
        val = (self.chunkCount - currLayer[0].counter) * self.chunkLen * self.tailsCo[layerNum]
        return (layerNum, 0, val)


    def latencyWinFirst(self, layerNum):
        '''

        :param layerNum - the layer that is being valued
        :return: the index of the first chunk in the latency window, if the chunk does not exist, return a negrive
        index instead
        '''
        exist = False
        resultChunk = None
        currLayer = self.streamBuffer[layerNum]
        if len(self.streamBuffer[layerNum]) == 0:
            return False, None

        if self.chunkCount < self.latencyWinSize:
            resultChunk = currLayer.popleft()
            exist = True
        else:
            idx = len(currLayer) - 1
            while idx >= 0 and currLayer[idx].counter > self.chunkCount - self.latencyWinSize:
                idx -= 1
            if currLayer[idx] == self.chunkCount - self.latencyWinSize + 1:
                exist = True
                resultChunk = currLayer[idx]
                del currLayer[idx]
                assert resultChunk is not None # Sanity Check
        return exist, resultChunk


class Plotter:
    '''
    Plotter - provide functions to plot needed plots for experiments,
    ie. video quality over time for archival video, viewer joins streaming after k_1 sec,
    after k_2 sec... and immediate viewer
    '''
    def __init__(self, outputBuffer, latency):
        '''
        Initialize the plotter
        :param outputBuffer: Stream.outputBuffer
        '''
        self.outputBuffer = outputBuffer
        self.latency = latency
        layerNum = outputBuffer.layerNum
        layerLen = len(outputBuffer[0])
        layerList = []
        for layer in range(layerNum):
            currList = list(outputBuffer[layer])
            currList.sort(key = lambda c: c.counter)
            layerList.append(currList)

        self.videoLen = layerLen # number of segments per layer
        self.sortedSegs = layerList
        self.layerNum = outputBuffer.layerNum
        # Step 1 sort the chunks according to Chunk.counter
        # For live user start right after time defined by Latency

    def avgQuality(self, delay = 0.0):
        cnt = 0
        qualityList = []
        sizeList = []
        while cnt < self.videoLen-1:
            for layer in range(self.layerNum):
                currQual = Quality([0.0], [0.0])
                currSize = 0.0
                currSeg = self.sortedSegs[layer][cnt]
                # pdb.set_trace()
                if currSeg.sentTime <= (cnt+1)*2.0 + delay:
                    currQual = currSeg.quality
                    currSize += currSeg.size
                else:
                    break
            qualityList.append(currQual)
            sizeList.append(currSize)
            cnt += 1

        # for exp test avg sizes for each layer
        l0 = [self.sortedSegs[0][i].size for i in range(299)]
        avg_0 = sum(l0)/len(l0)
        l1 = [self.sortedSegs[1][i].size for i in range(299)]
        avg_1 = sum(l1)/len(l1)
        l2 = [self.sortedSegs[2][i].size for i in range(299)]
        avg_2 = sum(l2)/len(l2)
        l3 = [self.sortedSegs[3][i].size for i in range(299)]
        avg_3 = sum(l3)/len(l3)
        pdb.set_trace()
        psnrSum = 0.0
        ssimSum = 0.0
        # calculate the average bit rate
        avgBitRate = sum(sizeList)*1.0/(len(sizeList)*2)
        print("avgBitRate: "+str(avgBitRate))

        for quality in qualityList:
            psnrSum += quality.psnr
            ssimSum += quality.ssim
        avg_psnr = psnrSum/len(qualityList)
        avg_ssim = ssimSum/len(qualityList)
        pdb.set_trace()
        return avg_psnr, avg_ssim

    def plotLiveUser(self):
        return


def main():
    bandWidths = BandWidth(1.0, [2000000] * 2000)
    streamChunks = loadSegments()
    stream = Stream(LAYERNUM, 8, streamChunks, bandWidths, None)
    stream.run()
    outputBuffer = stream.outputBuffer
    plotter = Plotter(outputBuffer, 8)
    plotter.avgQuality(299.0)

    pdb.set_trace()


def test():
    bandWidths = BandWidth(1.0, [3.0]*10000)
    streamChunks = Buffer(3)
    for i in range(100):
        streamChunks.addChunk(Chunk(1.0, 0, 1.0, 1.0))
        streamChunks.addChunk(Chunk(2.0, 1, 1.0, 1.0))
        streamChunks.addChunk(Chunk(2.0, 2, 1.0, 1.0))
    stream = Stream(3,2,streamChunks,bandWidths, None)
    stream.run()

if __name__ == "__main__":
    test()
    # print("testing")
    # streamChunks = Buffer(3)
    # totalLayer = 3
    # layer0Size = 100
    # layer1Size = 110
    # layer2Size = 120
    # for i in range(100):
    #     streamChunks.addChunk(Chunk(layer0Size, 0, layer0Size/2, 2)) #layer 0
    #     streamChunks.addChunk(Chunk(layer1Size, 1, (layer0Size + layer1Size) / 2, 2)) # layer 1
    #     streamChunks.addChunk(Chunk(layer2Size, 2, (layer0Size + layer1Size + layer2Size)/ 2, 2)) # layer 2
    #
    #
    # stream = Stream(3, 6, streamChunks, bandWidths, None)
    #
    # stream.run()
    #
    # pdb.set_trace()
    # print "testing"
    # streamChunks = Buffer(3)
    # totalLayer = 3
    # layer0Size = 52427*1000
    # layer1Size = 53656*1000
    # layer2Size = 53663*1000
    # for i in range(100):
    #     streamChunks.addChunk(Chunk(layer0Size, 0, layer0Size/2, 2)) #layer 0
    #     streamChunks.addChunk(Chunk(layer1Size, 1, (layer0Size + layer1Size) / 2, 2)) # layer 1
    #     streamChunks.addChunk(Chunk(layer2Size, 2, (layer0Size + layer1Size + layer2Size)/ 2, 2)) # layer 2
    #
    # bandWidths = BandWidth(1.0, [50000*1000*2] * 1000)
    #
    # #pdb.set_trace()
    # stream = Stream(3, 6, streamChunks, bandWidths, None)
    #
    # stream.run()
    # pdb.set_trace()


