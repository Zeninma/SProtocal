from collections import deque
import math

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
    bandWidthQueue = deque()
    def __init__(self, unit):
        '''

        :param unit: number of seconds each element represents
        '''
        self.unitTime = unit

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
        self.buffer = [deque()] * numLayer
        self.layerNum = numLayer

    def addChunk(self, chunk):
        layer = chunk.layer
        self.buffer[layer].append(chunk)

    def addStreamChunks(self, streamChunks, timeElapse):
        '''

        :param streamChunks - the buffer contains all the chunks that will be produced by video encoder
        :param timeElapse - number of seconds passed since the last addStreamChunks happened
        :return:
        '''
        totalSliceNum = timeElapse/streamChunks[0][0].timeLen
        for sliceNum in range(totalSliceNum):
            for layer in streamChunks.layerNum:
                newChunk = streamChunks[layer].popLeft()
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
       self.latencyWinSize = math.ceil(self.latency*1.0/self.chunkLen)
       self.initAlg()

    def initAlg(self):
        '''
        Initialize the parameters for the algorithm
        :return:
        '''


    def run(self):
        # the starting period, where self.latency seconds of video will be transmitted to the server
        self.streamBuffer.addStreamChunks(self.streamChunks, self.latency)

        while len(self.streamChunks[0]) != 0 or len(self.streamBuffer.empty()) != 0:
            if not self.streamBuffer.empty():
                currChunk = self.getNextChunk()
                self.send(currChunk)
            else:
                # @TODO: need to increment timer and add new chunks into stream buffer
                self.timeCounter = (math.floor(self.timeCounter/self.chunkLen)+1)*self.chunkLen
                self.streamBuffer.addStreamChunks(self.streamChunks, self.chunkLen)


    def send(self, chunk):
        '''
        calculate the time needed to send the chunk, increment the time counter,
        set the arrive time for chunk and put it into self.outputBuffer
        :param chunk - the chosen chunk to be sent to the server
        :return:
        '''
        startTime = self.timeCounter
        currBWIdx = self.timeCounter/self.chunkLen
        currBW = self.bwList[currBWIdx]
        size = chunk.size
        while True:
            if size - currBW * self.bwList.unitTime > 0: # determine whether to update current bandwidth
                currBWIdx += 1
                currBW = self.bwList[currBWIdx]
                size -= currBW * self.bwList.unitTime
                self.timeCounter += self.bwList.unitTime
            else:
                self.timeCounter += size*1.0/currBW
                chunk.setSent(self.timeCounter)
                self.outputBuffer.addChunk(chunk)
                # Add new chunk into streamBuffer
                # if stream chuncks is not empty
                if len(self.streamChunks[0]) > 0:
                    self.streamBuffer.addStreamChunks(self.streamChunks,self.timeCounter - startTime)
                return


    def getNextChunk(self):
        '''
        pop a chunk from self.streamBuffer to send to the server, using our algorithm
        :return: The chunk to send next
        '''

        return Chunk()

    def tailVal(self,layerNum):
        '''

        :param layerNum - the layer that is being valued
        :return: score for the tail chunk at the given layer
        '''
        if len(self.streamBuffer[layerNum]) == 0:
            return 0
        tailChunk = self.streamBuffer[layerNum][0]
        coeff = self.param.get("tail"+layerNum)
        return coeff*(self.timeCounter - tailChunk.birthTime)/tailChunk.size

    def latencyWinIdx(self, layerNum):
        '''

        :param layerNum - the layer that is being valued
        :return: the index of the first chunk in the latency window, if the chunk does not exist, return a negrive
        index instead
        '''
        targetBirth = math.floor((self.timeCounter - self.latency)/self.chunkLen) * self.chunkLen
        currLayer = self.streamBuffer[layerNum]
        searchRange = int(self.latency/self.chunkLen)
        #for i in range(searchRange):
        #   idx =

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

