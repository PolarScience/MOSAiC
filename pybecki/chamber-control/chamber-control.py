#!/home/venv/bin/python

import pygl
import socket
import sys
import argparse
import json
import subprocess

CMD_SHOW_NETWORK = {"network": "show"}
CMD_OPEN = {"cmd": "open"}
CMD_CLOSE = {"cmd": "close"}
CMD_STOP = {"cmd": "stop"}


class Animator(object):
    """
    This iterator provides an animation during chamber movement (similar to a progress bar).
    It indicates that the chamber is still moving and the onnection is alive.
    """
    def __init__(self, direction):
        """
        :param direction: str, either 'open' or  'close'
        """
        # string drawing the chamber frame with ascii characters spread over 6 lines in the terminal.
        # Since strings are immutable, we make a list of chars instead
        self.outlist = list('\\ \\\r\n \\ \\\r\n  \\ \\\r\n  | |\r\n  | |\r\n  | |\r\n')
        # index for each line, where the chamber lid resides
        self.indeces = [1, 7, 14, 21, 28, 35]

        # depending on the diretion, the chamber lid is animated accordingly
        # with dir_char, also a different character (e.g. arrows) can be used for each direction
        if direction == 'open':
            dir_mult = -1
            dir_off = 1
            dir_char = '*'
        elif direction == 'close':
            dir_mult = 1
            dir_off = 0
            dir_char = '*'
        else:
            raise RuntimeError("direction argument must be either 'open' or 'close'")
        self.dir_mult = dir_mult
        self.dir_off = dir_off
        self.dir_char = dir_char
        # count indicates the current line where the chamber lid is in the animation
        self.count = 0
        self.init = True

        # the following is a workaround that the animation also works under Windows 10
        subprocess.call('', shell=True)

    def __next__(self):
        """
        iterate the animated chamber lid. Move the lid one line up (open) or down (close).
        :return:
        """
        if not self.init:
            # move the cursror back to top (6 lines up), must not be done when the frame is printed first
            sys.stdout.write('\033[6A')
        else:
            self.init = False

        # first, print the frame with the chamber lid at the current position (count)
        finlist = self.outlist.copy()
        # insert the chamber lid at the desired position
        finlist[self.indeces[(self.count + self.dir_off) * self.dir_mult]] = self.dir_char
        # write the string (converted from lits) to the terminal (stdout)
        sys.stdout.write(''.join(finlist))
        # flush the output buffer (recommended)
        sys.stdout.flush()

        # second, proceed to next step
        if self.count == 5:
            self.count = 0
        else:
            self.count += 1

    def __iter__(self):
        return self

    @staticmethod
    def clean():
        """
        Delete the animation line by line
        :return:
        """
        for i in range(6):
            # move cursor one line up, then delete line
            sys.stdout.write('\033[1A\033[K')
            sys.stdout.flush()


def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser(description='User friendly chamber control.')
    parser.add_argument('operation', help="operation you want to perform with the chamber:"
                                          "status, open, close", type=str, default="status")
    parser.add_argument('ip', help="ip of the chamber or localhost if local server is running",
                        type=str, default="localhost")
    parser.add_argument('-p', '--port', help="tcp/ip port, defaults to 23", type=int)
    parser.add_argument('-t', '--timeout', help="certain operations require a timeout, defaults to 210 seconds",
                        type=int, default=210)

    args = parser.parse_args()

    # do we want to connect to a local server or to the chamber directly?
    # chamberlocalserver (on localhost) sends line with only \n as line ending. The readline() function (see below)
    # should actually cover all common line endings. For some reason, readline() only returns, when I manually
    # set newline to \n
    if (args.ip == 'localhost') or (args.ip == '127.0.0.1'):
        localhost = True
        newline = '\n'
    else:
        localhost = False
        newline = None
    # default port settings differ for local server or direct connection
    if args.port:
        port = args.port
    else:
        if localhost:
            port = 30000
        else:
            port = 23

    # logging logger predefined for housekeeping with grassland standards
    housekeeper = pygl.get_housekeeper()

    # connect to the chamber
    sock = socket.socket()
    sock.connect((args.ip, port))
    sock.settimeout(2)

    # if we connect directly to the chamber, we have to say hello with a "\r\n"
    if not localhost:
        sock.sendall(b'\r\n')

    # first, read the chamber's IP
    # When connected via chamberlocalserver on localhost, the actual IP of the chamber is not known.
    # Thus, we query the IP by command {"network": "show"}. This is already a connection test.

    # we read the first line and check if it contains network information
    with sock.makefile(newline=newline) as s:
        line = s.readline().rstrip('\r\n')
    # save the received line (JSON serialized) in a python dictionary
    linedict = dict()
    try:
        # under certain circumstances, the first line is not complete and will produce a json decode error
        linedict = json.loads(line)
    except json.decoder.JSONDecodeError:
        # but we simply ignore it
        pass

    try:
        # check if it contains network information
        networkdict = linedict['network']
    except KeyError:
        # if the received line does not contain network information, we have to query it:
        with sock.makefile(newline=newline) as s:
            sock.sendall(json.dumps(CMD_SHOW_NETWORK).encode() + b'\r\n')
            network = ""
            while "network\": {" not in network:
                network = s.readline().rstrip('\r\n')
        networkdict = json.loads(network)['network']
    else:
        with sock.makefile(newline=newline) as s:
            line = ''
            while line == '':
                line = s.readline().rstrip('\r\n')
        # save the received line (JSON serialized) in a python dictionary
        linedict = json.loads(line)

    # actual work to be done according to the operation (input argument)
    command = args.operation.lower()
    command_success = False
    if command == 'status':
        print("Limit switch active: {}".format(linedict['limits']))
        print("Movement: {}".format(linedict['movement']))
        print("Command: {}".format(linedict["cmd"]))
        print("Time elapsed since command issued: {}".format(linedict['t_elapsed']))
        print("Air temperature in chamber: {}".format(linedict['data']['TA']))
        print("Air pressure in chamber: {}".format(linedict['data']['PA']))
        print("Supply voltage: {}".format(linedict['data']['v_motor']))

    elif command == 'open' or command == 'close':
        limitdict = {'open': 'upper', 'close': 'lower'}
        json_command = json.dumps({'cmd': '{}'.format(command)}).encode()

        # check the current position of the chamber
        if linedict['limits'] == limitdict[command]:
            housekeeper.warning('Want to {} chamber {}, but is already {}ed'
                                .format(command, networkdict['ip'], command.rstrip('e')))
        elif linedict['limits'] == 'both':
            housekeeper.warning('Want to {} chamber {}, but both limit switches are active.'
                                .format(command, networkdict['ip']))
        else:
            animation = Animator(command)
            housekeeper.info('{} chamber {}'.format(command.capitalize(), networkdict['ip']))
            # send the command
            sock.sendall(json_command + b'\r\n')
            time_elapsed = 0
            # wait until the chamber has stopped
            try:
                while True:
                    if time_elapsed > args.timeout:
                        housekeeper.warning('Chamber {} has not {}ed within {} seconds. Timeout expired.'
                                            .format(networkdict['ip'], command.rstrip('e'), args.timeout))
                        break
                    with sock.makefile(newline=newline) as s:
                        line = s.readline()

                    linedict = json.loads(line)
                    try:
                        time_elapsed = linedict['t_elapsed']
                        if linedict['movement'] != "{}ing".format(command.rstrip('e')):
                            if linedict['limits'] == limitdict[command]:
                                animation.clean()
                                command_success = True
                                housekeeper.info('Chamber {} has {}ed and used {} seconds'
                                                 .format(networkdict['ip'], command.rstrip('e'), time_elapsed))
                                break
                            elif linedict['limits'] == 'both':
                                animation.clean()
                                housekeeper.critical('Chamber {} reports both limit switches to be active!'
                                                     .format(networkdict['ip']))
                                break
                            elif linedict['movement'] == 'motor-fault':
                                animation.clean()
                                housekeeper.warning('Chamber {} encountered a motor fault'
                                                    .format(networkdict['ip']))
                                break
                            elif linedict['cmd'] == 'error' or linedict['cmd'] == '':
                                pass
                            elif linedict['cmd'] != command:
                                animation.clean()
                                housekeeper.warning("Chamber {} {}ing has been cancelled by a '{}' command"
                                                    .format(networkdict['ip'], command.rstrip('e'), linedict['cmd']))
                                break
                    except KeyError:
                        pass
                    else:
                        next(animation)
            except KeyboardInterrupt:
                # stop the chamber movement
                sock.sendall(json.dumps(CMD_STOP).encode() + b'\r\n')
                housekeeper.warning("Chamber {} {}ing has been cancelled by keyboard interrupt".
                                    format(networkdict['ip'], command.rstrip('e')))
            except socket.timeout:
                housekeeper.warning("Connection to chamber {} lost.".format(networkdict['ip']))
                raise

    elif command == 'stop':
        sock.sendall(json.dumps(CMD_STOP).encode() + b'\r\n')
        housekeeper.info('Stop chamber {}'.format(networkdict['ip']))
        while True:
            with sock.makefile(newline=newline) as s:
                line = s.readline()
            linedict = json.loads(line)
            try:
                if linedict['movement'] == 'stopped':
                    command_success = True
                    housekeeper.info('Chamber {} has successfully stopped.'.format(networkdict['ip']))
                    break
            except KeyError:
                pass
    else:
        raise argparse.ArgumentError("operation argument not valid")

    # close socket
    sock.detach()
    sock.close()

    # exit according to success level
    if command_success:
        exit(0)
    else:
        exit(1)


# ******************************
if __name__ == "__main__":
    main(sys.argv[1:])