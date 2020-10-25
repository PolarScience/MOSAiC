# MecMast Suite
This repository contains the python scripts to adapt the Fireco MecMast to the 
snowheight. The script `mecmast.py` is used to control or move the mast and read its
status, whereas `adapt_mast.py` is used to measure the relative distance to the snow
surface and executes `mecmast.py` if necessary.

## Prerequisites
Python 3.6 or greater

### Python packages
* [pyserial-asyncio](https://pypi.org/project/pyserial-asyncio/)
* [pygl](https://gitlab.ethz.ch/gl/pygl)

### other
* a service running [shm50localserver](https://gitlab.ethz.ch/gl/shm50localserver)

## Install
* copy paste to `usr/local/bin` and grant executing permissions using `chmod`
* make sure the line ending fits your operating system (unix: lf, dos: crlf)

## Usage
Usually, you need sudo in order to access the housekeeping files
```
usage: adapt_mast.py [-h] [--loglevel LOGLEVEL] [--avg_time AVG_TIME]
                     [--threshold THRESHOLD] [--disable_slope]
                     host port

Control and the Fireco Mecmast.

positional arguments:
  host                  ip address where the distance sensor is serving
  port                  tcp port where the distance sensor is serving

optional arguments:
  -h, --help            show this help message and exit
  --loglevel LOGLEVEL   the logging level
  --avg_time AVG_TIME   averaging time in minutes
  --threshold THRESHOLD
                        absolute distance in mm, above which the mast is
                        adapted. Default is 50
  --disable_slope       disable the distance measurement correction due to the
                        slope angle
```
```
usage: mecmast.py [-h] [--loglevel LOGLEVEL] [job [job ...]]

Control and the Fireco Mecmast.

positional arguments:
  job                  Tell the program what to do. Choose from ['NO_COMMAND',
                       'STOP', 'UP', 'DOWN', 'AUTO', 'UNLOCK', 'ALARM',
                       'RESET', 'STATUS']. For 'auto' also provide the target
                       height in mm or relative change with +/- mm

optional arguments:
  -h, --help           show this help message and exit
  --loglevel LOGLEVEL  the logging level
```

## Description
###`mecmast.py`:
The mast acts as a master and continuously sends request packets on its busses
(all serial connections) for all possible slaves several times per second.
The packets differ slightly for the various slaves, but all start with a `STX`
followed by a one-byte address and end with en `EOT`. In between is the payload
and a checksum.

We are the slave named *PC* and the one-byte slave address for *PC* is `0x33`.
The whole message for *PC* has 63 bytes. That's the pattern we are searching for.

Once we have received a message, we must answer not before 10 ms after reception but
not later than 40 ms after the packet was sent by the mast(er). Depending on the
information in the message from the mast(er), we can compose a response which
contains commands for the mast.

Commands that move the mast have to be sent continuously during the whole
moving period. Continuously here means 10 ms after the reception of each master
request message.

However, all other commands have to be sent only once to take effect.

We make use of the python's asynchronous capabilities `asyncio` and the
module for async serial communication `pyserial_asyncio`. There we use the
`asyncio.Protocol`'s `data_received()` callback to decide if there is a message for us
(*PC*) and if so, if it has completed. Depending on the mast's status and the job we 
want to do with it (specified by commandline argument), we await a coroutine in 
which the corresponding response and potential further actions are done.

Since we use the `loop.run_forever()` method to run the asyncio event loop, we have to
terminate program by raising an exception. The `exception_handler` on the loop assures
that these exceptions are propagated to and caught by the event loop. Without this, 
it is possible that exceptions from scheduled tasks on the event loop are not caught.

###`adapt_mast.py`:
We read the distance to the (snow) surface from the SHM50 sensor. The SHM50 sensor data
are provided in JSON serialized format on a tcp socket (local server). The host and
port have to be given as arguments. Since the SHM50 delivers 3 independent distances,
we have to spatially average them in a first step. Basic quality control (no error, 
minimal signal intensity) filters out unreliable distance values. In a second step,
we temporally average over 5 min (default, can by adjusted by optional argument).

Since the sensor is pointing on a slightly tilted area (about 10°), we correct for that slope.

If the spatially and temporally averaged distance value is above a certain threshold,
the mast height will be adjusted by calling `mecmast.py` with corresponding arguments.

We use python's `asyncio` framework for the socket communication and the subprocess
execution. The tcp client runs and reads distance data until it is cancelled. The 
cancelling is done by the `asyncio.wait_for(task, timeout)` wrapper, which waits for
a task until timeout has passed.



