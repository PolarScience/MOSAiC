""" 
AUTHOR: Patrick Koller 

This routine is used to read from a serial port and split the string by a start or stop sequence 
USE: 
s = Serial('/dev/ttyS0')
len_line = s.readline(stopseq='\x10\x13')

len_line is a list with the number of chars read as first and the string as 2nd element
"""
"""Import modules:"""
import sys
import glob
import serial
#import pdb

"""Serial: Funktion"""
class Serial(serial.Serial):
   @staticmethod
   def listports():
      """ Lists serial port names
      :raises EnvironmentError:
      On unsupported or unknown platforms
      :returns:
      A list of the serial ports available on the system
      """

      if sys.platform.startswith('win'):
         ports = ['COM%s' % (i + 1) for i in range(256)]
         print('Port: ', ports)

      elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
         print('Serialio: WE RUN LINUX')
         # this excludes your current terminal "/dev/tty"
         ports = glob.glob('/dev/tty[A-z][0-9]*')
         print('Serialio: checkports: ', ports)
      elif sys.platform.startswith('darwin'):
         print('Serialio: WE RUN darwin')
         ports = glob.glob('/dev/tty.*')
      else:
         raise EnvironmentError('Unsupported platform')

      result = []
      for port in ports:
         try:
            s = Serial(port)
            s.close() #objekt.methode
            result.append(port)
         except (OSError,serial.SerialException):
            pass
      return result #available com ports

   #C:/Users/becki/Desktop/AWS MecMast/PY_AWS_Debug/pyprogs/serialio/serialio.py #only for PC at LFW (test scenario)
   def readmsg(self,startseq=b'',stopseq=b'',bytestoread = -1):
      hasstartseq = len(startseq) >= 1
      hasstopseq = len(stopseq) >= 1
      hasbytestoread = bytestoread >= 0

      if hasstartseq and hasstopseq and not hasbytestoread:
         nbytes0 = max(1,len(startseq))
         nbytes1 = max(1,len(stopseq))
      elif hasstartseq and not hasstopseq and not hasbytestoread:
         stopseq = startseq
         nbytes0 = max(1,len(startseq))
         nbytes1 = max(1,len(stopseq))
      elif not hasstartseq and hasstopseq and not hasbytestoread:
         startseq = stopseq
         nbytes0 = max(1,len(startseq))
         nbytes1 = max(1,len(stopseq))
      elif hasstartseq and not hasstopseq and hasbytestoread:
         nbytes0 = max(1,len(startseq))
         nbytes1 = max(1,len(stopseq))
      else:
         raise RuntimeError('ERROR: Neither startseq nor stopseq was provided!')

      line = b''
      hasstarted = False 
      while True:
         #pdb.set_trace()
         line += self.read(1)
         #print("line: ", line)
         """ If the end of the line matches the startseq delete everything before the start sequence """
         if not hasstarted and line[-min(nbytes0,len(line)):] == startseq:
            hasstarted = True
            #pdb.set_trace()
            if hasstartseq and not hasbytestoread:
               line = line[-min(nbytes0,len(line)):]
            elif hasstartseq and hasbytestoread:
               line = line[-min(nbytes0,len(line)):]
               line += self.read(bytestoread - nbytes0)
               #pdb.set_trace()
               return [len(line), line]

            else:
               line = b''
            continue

         """ If the end of the line matches the stopseq return the sequence """
         if hasstarted and line[-min(nbytes1,len(line)):] == stopseq:
            hasstarted = False
            if hasstopseq:
               return [len(line), line]
            else:
               line = line[0:-nbytes1]
               return [len(line), line]


               # if __name__ == '__main__':
#    ser = Serial('/dev/t1mac')
#    #print(ser.listports())
#    #pdb.set_trace()
#    for ii in range(0,100):
#       line = ser.readmsg(startseq=b'\x04\x02',stopseq=b'',bytestoread=-1)
#       #pdb.set_trace()
#       print('Read %02i chars: %s ' % (line[0],line[1].hex()))