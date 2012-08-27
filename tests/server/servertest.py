# Quick calibrig server test
from socket import *

#
host = "localhost"
port = 8091

s = socket(AF_INET,SOCK_DGRAM)
s.connect((host, port))
s.sendto("MAXPOINTS 500\n".ljust(128), (host,port))
s.settimeout(0.1)
try:
    data = s.recv(2048)
    print data
except timeout:
    print "timeout"

s.close()

