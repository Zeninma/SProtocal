import Queue

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
    bandWidthQueue = Queue()
    def __init__(self, unit):
        '''

        :param unit: number of seconds each element in the bandWidthQueue represents
        '''
        self.unitTime = unit

class Chunk:
    '''
    Chunk -- represents a chunk of video with fixed length
    '''
    size = 0
    layer = 0
    quality = 0
    timeLen = 0.0
    arriveTime = 0.0 # This is set when the packet is sent to the server
    def __init__(self, newSize, newLayer, newQuality, newTimeLen):
        '''

        :param newSize: size of the chunk in Mb
        :param newLayer: the layer this chunk belongs to
        :param newQuality: quality of the chunk, might be an object wraps several
        metrics for quality measurement
        :param newTimeLen: number of second of video this chunk contains
        '''
        self.size = newSize
        self.layer = newLayer
        self.quality = newQuality
        self.timeLen = newTimeLen

class Buffer:
    '''
    Buffer -- represents the uploader's in memory buffer,
    which holds encoded video chunks
    '''
    layerNum = 0
    buffer = []
    def __init__(self, numLayer):
        self.buffer = [Queue()] * numLayer
        self.layerNum = numLayer

class Stream:
    '''
    Stream - simulate the stream process
    '''
    numLayer = 0
    latency = 0.0
    chunkGenTime = 0.0
    timeCouter = 0.0

    def __init__(self, numLayer, latency, chuckGenTime, streamChuncks, bwList):
       '''

       :param numLayer: number of layers produced by the encoder
       :param latency: latency for streaming (around 10 sec is acceptable)
       :param chuckGenRate: number of sec needed to generate layer0 to numLayer-1 chunks
       :param streamChuncks: chunks for upload wrapped as a Buffer object
       :param bwList: The bandwidth for uploading
       '''
       self.numLayer = numLayer
       self.latency = latency
       self.chunkGenTime = chuckGenTime
       self.streamChuncks = streamChuncks
       self.bwList = bwList
       self.streamBuffer = Buffer(self.numLayer)
       self.outputBuffer = Buffer(self.numLayer) # represent the chunks received by the server

    def run(self):
        while(len(self.streamChuncks[0])==0 and len(self.streamBuffer[0])==0):
            currChunk = self.value()


    def send(self, chunk):
        '''
        calculate the time needed to send the chunk, increment the time counter,
        set the arrive time for chunk and put it into self.outputBuffer
        :param chunk: the chosen chunk to be sent to the server
        :return:
        '''

    def value(self):
        '''
        pop a chunk from self.streamBuffer to send to the server
        :return: The chunk popped
        '''
        return Chunk()

class Plotter:
    '''
    Plotter - provide functions to plot needed plots for experiments,
    ie. video quality over time for archival video, viewer joins streaming after k_1 sec,
    after k_2 sec... immediate viewer
    '''
    def plot(self, joinTime):
        return
