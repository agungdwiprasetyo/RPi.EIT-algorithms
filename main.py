#!/usr/bin/env python3
from run import run
from config import *
from socketIO_client import SocketIO, LoggingNamespace

socketIO = SocketIO(host, port, LoggingNamespace)

# main function
if __name__ == '__main__':
	print ("Listening server...")
	socketIO.on('startReconstruction', run)
	socketIO.wait()