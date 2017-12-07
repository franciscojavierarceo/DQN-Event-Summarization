import glob
import os
import struct
from nltk.tokenize import sent_tokenize

from tensorflow.core.example import example_pb2

def getsentences(file0):
    reader = open(file0, 'rb')
    len_bytes = reader.read(8)
    str_len = struct.unpack('q', len_bytes)[0]
    example_str = struct.unpack("%ds" % str_len, reader.read(str_len))[0]
    content = example_pb2.Example.FromString(example_str)

    article  = str(content.features.feature['article'].bytes_list.value[0])
    abstract = str(content.features.feature['abstract'].bytes_list.value[0])

    return sent_tokenize(abstract), sent_tokenize(article)

def main():
    file0 = '/home/francisco/GitHub/cnn-dailymail/finished_files/chunked/train_000.bin'
    return getsentences(file0)

if __name__ == '__main__':
        print(main())
