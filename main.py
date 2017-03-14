#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function
from run import run
from socketIO_client import SocketIO, LoggingNamespace

host = 'localhost'
port = 3456

socketIO = SocketIO(host, port, LoggingNamespace)
socketIO.emit('raspiConnect', {'status': True})


# main function
if __name__ == '__main__':
	print ("Listening server...")
	socketIO.on('startReconstruction', run)
	socketIO.wait()