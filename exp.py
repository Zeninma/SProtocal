from collections import deque
import math
import pdb, traceback, sys

# @TODO: write a function to read a csv file of chunks. Every chunk will be converted to a Chunk object,
# and all the Chunk objects will be wrapped into a Buffer Object.
# @TODO: write a function to read a csv file of upload bandwidth trace and return a BandWidth object

class Quality:
    '''
    Quality -- a wrapper object containing metrics that are used to represent the quality
    of the video
    '''
    quality = {}

class BandWidth:
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
    counter = 0 # indicate which round the chunk is added to the stream.streamBuffer
    size = 0 # the number of bytes of the content
    layer = 0 # which layer does this chunk belong to
    timeLen = 0.0 # time length for each content
    sentTime = 0.0 # This is set when the packet is sent to the server

    def __init__(self, newSize, newLayer, newQuality, newTimeLen):
        '''

        :param newSize: size of the chunk in Mb
        :param newLayer: the layer this chunk belongs to
        :param newQuality: quality of the chunk, might be an object wraps several
        metrics for quality measurement wrapped as a Quality object
        :param newTimeLen: number of second of video this chunk contains
        '''
        self.size = newSize
        self.layer = newLayer
        self.quality = newQuality
        self.timeLen = newTimeLen

    def setSent(self, sentTime):
        self.sentTime = sentTime

    def setCounter(self, counter):
        self.counter = counter

class Buffer:
    '''
    Buffer -- represents the uploader's in memory buffer,
    which holds encoded video chunks
    '''
    layerNum = 3
    buffer = []

    def __init__(self, numLayer = 3):
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
        :return:
        '''
        for layer in range(streamChunks.layerNum):
            for sliceNum in range(totalSliceNum):
                newChunk = streamChunks[layer].popleft()
                # try:
                #     newChunk = streamChunks[layer].popleft()
                # except:
                #     type, value, tb = sys.exc_info()
                #     traceback.print_exc()
                #     pdb.post_mortem(tb)
                newChunk.setCounter(chunkCount + sliceNum + 1)
                self.buffer[layer].append(newChunk)

    def empty(self):
        result = True
        for layer in range(self.layerNum):
            result = result and len(self.buffer[layer]) == 0
        return result


class Stream:
    '''
    Stream - simulate the stream process
    '''
    numLayer = 0
    latency = 0.0
    chunkGenTime = 0.0
    timeCounter = 0.0
    chunkCount = 0 # Indicating which round the chunk at the head of streamBuffer is added


    def __init__(self, numLayer, latency, streamChunks, bwList, algParam):
        '''

        :param numLayer - number of layers produced by the encoder
        :param latency - latency for streaming (around 10 sec is acceptable)
        :param streamChunks - chunks for upload wrapped as a Buffer object
        :param bwList - The bandwidth for uploading
        :param algParam - a dictionary for algorithm's parameters
        '''
        self.numLayer = numLayer
        self.latency = latency
        self.streamChunks = streamChunks
        self.bwList = bwList
        self.streamBuffer = Buffer(self.numLayer) # representing chunks in memory buffer at self.timeCounter
        self.outputBuffer = Buffer(self.numLayer) # represent the chunks received by the server
        self.chunkLen = streamChunks[0][0].timeLen
        self.latencyWinSize = int(math.ceil(self.latency*1.0/self.chunkLen))
        self.initAlg()
        # first fill the buffer



    def initAlg(self):
        '''
        @TODO: modify to take a real algParam
        Initialize the coefficient
        :return:
        '''
        self.headsCo = [40,20,10]
        self.tailsCo = [4,2,1]


    def run(self):
        # the starting period, where self.latency seconds of video will be transmitted to the server
        self.streamBuffer.addStreamChunks(self.streamChunks, self.latencyWinSize, self.chunkCount)
        #pdb.set_trace()

        while not self.streamChunks.empty() or not self.streamBuffer.empty():
            if not self.streamBuffer.empty():
                currChunk = self.getNextChunk()
                self.send(currChunk)
            else:
                # @TODO: need to increment timer and add new chunks into stream buffer
                self.timeCounter = (math.floor(self.timeCounter/self.chunkLen)+1)*self.chunkLen
                self.streamBuffer.addStreamChunks(self.streamChunks, 1, self.chunkCount)
                self.chunkCount += 1


    def send(self, chunk):
        '''
        calculate the time needed to send the chunk, increment the time counter,
        set the arrive time for chunk and put it into self.outputBuffer
        :param chunk - the chosen chunk to be sent to the server
        :return:
        '''
        startTime = self.timeCounter
        currBWIdx = int(self.timeCounter/self.bwList.unitTime)
        timeLeft = (currBWIdx+1) * self.bwList.unitTime - self.timeCounter
        currBW = self.bwList[currBWIdx]
        size = chunk.size
        while True:
            if size - currBW * timeLeft > 0: # determine whether to update current bandwidth
                size -= currBW * timeLeft
                self.timeCounter += timeLeft
                currBWIdx += 1
                print currBWIdx
                currBW = self.bwList[currBWIdx]
                timeLeft = self.bwList.unitTime
            else:
                self.timeCounter += size*1.0/currBW
                chunk.setSent(self.timeCounter)
                self.outputBuffer.addChunk(chunk)
                # Add new chunk into streamBuffer
                # if stream chuncks is not empty
                timeElapsed = self.timeCounter - startTime
                sliceNum = int(timeElapsed/self.chunkLen)
                if len(self.streamChunks[0]) > 0:
                    sliceNum = min(sliceNum, len(streamChunks[0]))
                    self.streamBuffer.addStreamChunks(self.streamChunks,sliceNum, self.chunkCount)
                    self.chunkCount += sliceNum
                return

    def getNextChunk(self):
        '''
        Use our algorithm to determine the next chunk to send
        :return: The chunk to send next
        '''
        for layer in range(self.numLayer):
            exist, result = self.latencyWinFirst(layer)
            if exist:
                return result

        heads = []
        tails = []

        for layer in range(self.numLayer):
            heads.append(self.getHead(layer))
            tails.append(self.getTail(layer))
        minPair = heads[0]

        for pair in heads:
            if pair[2] > minPair:
                minPair = pair
        for pair in tails:
            if pair[2] > minPair:
                minPair = pair
        resultChunk = self.streamBuffer[minPair[0]][minPair[1]]
        del self.streamBuffer[minPair[0]][minPair[1]]

        assert resultChunk is not None # Sanity Check

        return resultChunk


    def getHead(self, layerNum):
        currLayer = self.streamBuffer[layerNum]
        idx = len(currLayer) - 1
        while idx >= 0 and currLayer[idx] > self.chunkCount - self.latencyWinSize + 1:
            idx -= 1
        val = self.headsCo[layerNum]
        return (layerNum, idx, val)

    def getTail(self,layerNum):
        currLayer = self.streamBuffer[layerNum]
        val = (self.chunkCount - currLayer[0].counter) * self.tailsCo[layerNum]
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
    def plot(self, outputBuffer ,joinTime):
        return

if __name__ == "__main__":
    print "testing"
    streamChunks = Buffer(3)
    totalLayer = 3
    layer0Size = 52427*1000
    layer1Size = 53656*1000
    layer2Size = 53663*1000
    for i in range(100):
        streamChunks.addChunk(Chunk(layer0Size, 0, layer0Size/2, 2)) #layer 0
        streamChunks.addChunk(Chunk(layer1Size, 1, (layer0Size + layer1Size) / 2, 2)) # layer 1
        streamChunks.addChunk(Chunk(layer2Size, 2, (layer0Size + layer1Size + layer2Size)/ 2, 2)) # layer 2

    bandWidths = BandWidth(1.0, [50000*1000*2] * 1000)

    #pdb.set_trace()
    stream = Stream(3, 6, streamChunks, bandWidths, None)

    stream.run()
    pdb.set_trace()


