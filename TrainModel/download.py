import urllib2
import os.path
import numpy as np

baseURL = "http://www.hesse.io/dnn/";

def downloadFile(url):
    fileName = url.split('/')[-1];
    openUrl = urllib2.urlopen(url);
    openFile = open(fileName, 'wb');
    meta = openUrl.info();
    fullFileSize = int(meta.getheaders("Content-Length")[0]);
    print("Downloading: %s Bytes: %s" % (fileName, fullFileSize));

    fileSizeDownloaded = 0;
    blockSize = 8192;
    while True:
        buf = openUrl.read(blockSize);
        if not buf:
            break;
        fileSizeDownloaded += len(buf);
        openFile.write(buf);
        status = r"%10d  [%3.2f%%]" % (fileSizeDownloaded, fileSizeDownloaded * 100. / fullFileSize);
        status = status + chr(8)*(len(status)+1);
        print status + "    \r",;

    openFile.close()

def getFile(filename):
    if os.path.exists('./' + filename):
        print(filename + ' exists');
        return np.load('./' + filename);
    else:
        print(filename + ' does not exist, downloading file');
        downloadFile(baseURL + filename);
        print('finished downloading ' + filename);
        return np.load('./' + filename);
