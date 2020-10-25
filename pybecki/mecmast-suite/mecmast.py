#!/home/venv/bin/python

import asyncio
import sys
import serial_asyncio
import pygl
import argparse

housekeeper = pygl.get_housekeeper(group='AdaptiveMast')

CYCLE_STATUS = {0: 'waiting for button release',
                1: 'waiting for command',
                2: 'manual movement down',
                3: 'manual movement up',
                4: 'automatic movement',
                5: 'encoder set to zero',
                6: 'encoder set to max',
                9: 'alarm',
                10: 'movement alarm',
                18: 'max height set',
                20: 'waiting for unlock'
                }

ALARMS = {0b1: 'extension motor over-temperature',
          0b10: 'extension motor over-current',
          0b100: 'rotation motor over-temperature',
          0b1000: 'rotation motor over-current',
          0b10000: 'hardware shortcut',
          0b100000: 'encoder anomaly',
          0b1000000: 'potentiometer anomaly',
          0b10000000: 'connection anomaly'
          }

COMMANDS = {'NO_COMMAND': 0xff,
            'STOP': 0x00,
            'UP': 0x01,
            'DOWN': 0x02,
            'AUTO': 0x03,
            'BRAKE_UNLOCK': 0x06,
            'MOTOR_UNLOCK': 0x22,
            'ALARM_RESET': 0x5a,
            'RESET': 0xaa,
            'BOOT': 0xbc
            }


class SystemExitStop(SystemExit):
    pass


def bytes2int(byte_msg):
    """
    Convert a bytearray to an integer number in MSB first order

    :param byte_msg: bytearray to convert to integer
    :type byte_msg: bytearray
    :return: calculated value
    :rtype: int
    """
    value = 0
    msg_length = len(byte_msg)
    for i, char in enumerate(byte_msg):
        value |= char << (8*(msg_length-i-1))
    return value


def byte2alarms(alarm_byte):
    alarms = list()
    for alarm in ALARMS:
        if alarm & alarm_byte:
            alarms.append(ALARMS[alarm])
    return alarms


def parse_mecmast_message(msg):
    """
    Parse the bytearray received from the MecMast main board to a python dictionary

    :param msg: full message received from the MecMast main board
    :type msg: bytes
    :return: parsed dictionary
    :rtype: dict
    """
    mecmast_dict = dict()
    payload = msg[2:61]
    mecmast_dict.update({'VERSION_MAJOR': payload[0], 'VERSION_MINOR': payload[1]})
    mecmast_dict.update({'STARTS_UP_EXT': bytes2int(payload[2:5]), 'STARTS_DOWN_EXT': bytes2int(payload[5:8])})
    mecmast_dict.update({'WORKING_TIME_EXT': bytes2int(payload[8:11])})
    mecmast_dict.update({'CURRENT_EXT': bytes2int(payload[11:13])})
    mecmast_dict.update({'STARTS_UP_ROT': bytes2int(payload[13:16]), 'STARTS_DOWN_ROT': bytes2int(payload[16:19])})
    mecmast_dict.update({'WORKING_TIME_ROT': bytes2int(payload[19:22])})
    mecmast_dict.update({'CURRENT_ROT': bytes2int(payload[22:24])})
    mecmast_dict.update({'OUTPUT_STATUS': f"{payload[24]:08b}"})
    mecmast_dict.update({'INPUT_STATUS': f"{payload[25]:08b}"})
    mecmast_dict.update({'POTENTIOMETER': bytes2int(payload[26:28])})
    mecmast_dict.update({'ENCODER_MM': bytes2int(payload[28:31])})
    mecmast_dict.update({'RADIO_SIGNAL_LEVEL': payload[31]})
    #mecmast_dict.update({'ALARMS': f"{payload[32]:08b}"})
    mecmast_dict.update({'ALARMS': byte2alarms(payload[32])})
    mecmast_dict.update({'MAX_HEIGHT': bytes2int(payload[34:36])})
    mecmast_dict.update({'MIN_ANGLE': bytes2int(payload[36:38])})
    mecmast_dict.update({'MAX_ANGLE': bytes2int(payload[38:40])})
    mecmast_dict.update({'SCALE_CONSTANT': bytes2int(payload[40:42])})
    mecmast_dict.update({'HEIGHT_FROM_GROUND': bytes2int(payload[42:44])})
    mecmast_dict.update({'MAX_SHOWN_ANGLE': bytes2int(payload[44:46])})
    mecmast_dict.update({'PRESET_1': bytes2int(payload[46:48])})
    mecmast_dict.update({'PRESET_2': bytes2int(payload[48:50])})
    mecmast_dict.update({'PRESET_3': bytes2int(payload[50:52])})
    mecmast_dict.update({'PRESET_4': bytes2int(payload[52:54])})
    mecmast_dict.update({'PRESET_5': bytes2int(payload[54:56])})
    mecmast_dict.update({'CYCLE_STATUS_EXT': CYCLE_STATUS[payload[56]]})
    mecmast_dict.update({'CYCLE_STATUS_ROT': CYCLE_STATUS[payload[57]]})
    mecmast_dict.update({'CYCLE_STATUS': CYCLE_STATUS[payload[58]]})

    crc = sum(payload) & 0xff
    housekeeper.debug('Payload: {}'.format(mecmast_dict))
    housekeeper.debug('CRC calculated: {}, CRC read: {}'.format(crc, msg[-2]))

    return mecmast_dict


class MecmastProtocol(asyncio.Protocol):
    def __init__(self, pc_address, message_length, job, loop):
        """
        :param pc_address: address, that is used to communicate with the PC, usually 0x33
        :type pc_address: int
        :param message_length: expected full message length of the message addressed to the PC
        :type message_length: int
        :param job: command line arguments which indicate the job this script shall do
        :type job: list
        :param loop: asyncio event loop
        """
        self.transport = None
        self.data = b''
        self.address = pc_address
        self.message_lenght = message_length
        self.receiving_message = False
        self.job = job
        self.loop = loop
        self.initial_position = None
        self.mast_task = None
        self.communication_task = None
        self.alarm = False

    def connection_made(self, transport):
        self.transport = transport
        self.communication_task = self.loop.create_task(self.check_mast_communication())

    def data_received(self, data):
        # cancel the communication task. Only done once after self.connection_made
        if self.communication_task is not None:
            self.communication_task.cancel()
            self.communication_task = None

        # check if the mast communication control task has finished with exception
        if self.mast_task:
            if self.mast_task.exception() is not None:
                raise SystemExit

        # check if the data from the mecmast is for us
        if data[0] == 0x02:     # STX
            if data[1] == self.address:
                self.receiving_message = True

        # append data, if we are receiving a message
        if self.receiving_message:
            self.data += data

        # check if the received message has completed
        if len(self.data) == self.message_lenght:
            if self.data[self.message_lenght-1] == 0x04:    # EOT
                mecmast_dict = parse_mecmast_message(self.data)
                self.data = b''
                self.receiving_message = False

                # check, if we got an alarm
                if mecmast_dict['ALARMS'] or mecmast_dict['CYCLE_STATUS_EXT'] == CYCLE_STATUS[9]:
                    # if so, report it, reset the alarm and try to continue once
                    if self.alarm:
                        # we already tried to reset it. Apparently unsusscessful, thus terminate:
                        self.loop.create_task(self.control_mast(mecmast_dict, job=['STOP']))
                        housekeeper.critical("Resetting the alarms did not solve the problem!")
                    else:
                        self.alarm = True
                        housekeeper.warning("Got the following alarms: {}. Try to reset and continue"
                                            .format(mecmast_dict['ALARMS']))
                        self.loop.create_task(self.control_mast(mecmast_dict, job=['ALARM_RESET']))
                else:
                    if self.initial_position is None:
                        self.initial_position = mecmast_dict['ENCODER_MM']
                        if self.job[0] != 'STATUS' and self.job[1] != 'TERMINATE':
                            housekeeper.info('Mast is initially at {} mm from ground'.format(self.initial_position))

                    if self.job[0] == 'UP' or self.job[0] == 'DOWN' or self.job[0] == 'AUTO':
                        if mecmast_dict['CYCLE_STATUS_EXT'] == CYCLE_STATUS[0]:
                            # in case of 'wait button release', the automatic movement has finished.
                            # We can terminate here!
                            housekeeper.debug('waiting for button release')
                            self.loop.create_task(self.control_mast(mecmast_dict, job=['STOP']))
                        elif mecmast_dict['CYCLE_STATUS_EXT'] == CYCLE_STATUS[20]:
                            # in case of 'wait for unlock', unlock the motors and continue
                            housekeeper.debug('waiting for unlock')
                            self.loop.create_task(self.control_mast(mecmast_dict, job=['MOTOR_UNLOCK']))
                        else:
                            # do the job, schedule it to the event loop
                            self.loop.create_task(self.control_mast(mecmast_dict))
                    else:
                        # do the job, schedule it to the event loop
                        self.loop.create_task(self.control_mast(mecmast_dict))

    @staticmethod
    async def check_mast_communication(timeout=1):
        """
        If this coroutine is not cancelled within timeout, it will exit. This is to ensure, the program exits, when
        the mast is powered off or disconnected. The coroutine is scheduled in a task in self.connection_made and
        cancelled in self.data_received

        :param timeout: timeout in seconds
        :type timeout: int
        """
        try:
            await asyncio.sleep(timeout)
            housekeeper.critical('Mast seems to be disconnected or powered off!')
            raise SystemExit
        except asyncio.CancelledError:
            housekeeper.debug('Communication with mast established.')

    async def control_mast(self, mecmast_dict, job=None):
        """
        Coroutine method to tell the mast, what to do if necessary.

        :param mecmast_dict: parsed data from the MecMast main board
        :param job: job to execute, must be one of COMMANDS or 'STATUS'
        :type job: list
        :type mecmast_dict: dict
        """

        if job is None:
            job = self.job

        if job[0] == 'STATUS':
            self.transport.close()
            raise SystemExit
        else:
            await asyncio.sleep(0.01)
            if job[0] == 'UP':
                response = compose_response_pack('UP')
                self.transport.write(response)
            elif job[0] == 'DOWN':
                response = compose_response_pack('DOWN')
                self.transport.write(response)
            elif job[0] == 'AUTO':
                if '+' in job[1] or '-' in job[1]:
                    position = self.initial_position + int(float(job[1]))
                else:
                    position = int(float(job[1]))
                response = compose_response_pack('AUTO', position=position)
                self.transport.write(response)
            elif job[0] == 'STOP':
                # only write the stop command if necessary!
                if mecmast_dict['CYCLE_STATUS_EXT'] != CYCLE_STATUS[1]:
                    response = compose_response_pack('STOP')
                    self.transport.write(response)
                # report the final position, if the mast has moved
                if self.initial_position != mecmast_dict['ENCODER_MM']:
                    if self.job[1] != 'TERMINATE':
                        housekeeper.info('Mast is finally at {} mm from ground'.format(mecmast_dict['ENCODER_MM']))
                    self.transport.close()
                    raise SystemExit
            elif job[0] == 'MOTOR_UNLOCK':
                housekeeper.info('Motor unlock')
                response = compose_response_pack('MOTOR_UNLOCK')
                self.transport.write(response)
                if self.job[0] == job[0]:
                    self.transport.close()
                    raise SystemExitStop
            elif job[0] == 'BRAKE_UNLOCK':
                housekeeper.info('Brake unlock')
                response = compose_response_pack('BRAKE_UNLOCK')
                self.transport.write(response)
                if self.job[0] == job[0]:
                    self.transport.close()
                    raise SystemExitStop
            elif job[0] == 'BOOT':
                housekeeper.info('Boot command')
                response = compose_response_pack('BOOT')
                self.transport.write(response)
                if self.job[0] == job[0]:
                    self.transport.close()
                    raise SystemExitStop
            elif job[0] == 'ALARM_RESET':
                housekeeper.info('Alarm recognition')
                response = compose_response_pack('ALARM_RESET')
                self.transport.write(response)
                if self.job[0] == job[0]:
                    self.transport.close()
                    raise SystemExitStop
            else:
                housekeeper.info('Job "{}" unknown'.format(self.job[0]))
                self.transport.close()
                raise SystemExit


def compose_response_pack(command, position=None):
    """
    Compose the answer to the MecMast main board (called response pack)

    :param command: the command to issue
    :type command: str
    :param position: target height position
    :type position: int
    :return: response pack to be sent to the MecMast main board
    :rtype: bytearray
    """
    response = bytearray()
    response += b'\x02\x33'     # STX ADDRESS

    response.append(COMMANDS[command])  # MOT1 COMMAND

    # MOT1 POSITION
    if position is None:
        response += bytes(3)
    else:
        # translate int to bytearray of length 3
        for i in range(3):
            response.append((position & (0xff << 8*(2-i))) >> 8*(2-i))

    # MOT2 COMMAND
    if 'RESET' in command or 'UNLOCK' in command:
        response.append(COMMANDS[command])
    else:
        response.append(0xff)

    response += bytes(2)    # POSITION (not used)
    response.append(0x04)   # OUTPUT (0x04: Low Temperature enabled)

    response.append(sum(response[2:]) & 0xff)   # CRC
    response += b'\x04'     # EOT

    housekeeper.debug("Composed response: {}".format(response.hex()))
    return response


def exception_handler(loop, context):
    if isinstance(context, SystemExit) or isinstance(context, SystemExitStop):
        raise context


async def stop():
    """
    This coroutine is only executed, when the program wants to terminate. It assures that a stop command is issued.
    
    """
    housekeeper.debug("Issue a 'STOP' command")
    loop = asyncio.get_event_loop()
    protocol = MecmastProtocol(0x33, 63, ['STOP', 'TERMINATE'], loop)
    await serial_asyncio.create_serial_connection(loop, lambda: protocol, '/dev/mecmast', baudrate=9600)
    await asyncio.sleep(1)
    loop.stop()


def main(argv):
    # parse arguments
    parser = argparse.ArgumentParser(description='Control and the Fireco Mecmast.')
    parser.add_argument('--loglevel', help="the logging level", type=str, default="WARNING")
    parser.add_argument('job', help="Tell the program what to do. Choose from {}. "
                                    "For 'auto' also provide the target height in mm "
                                    "or relative change with +/- mm".format(list(COMMANDS.keys())+['STATUS']),
                        # choices=list(COMMANDS.keys()),
                        nargs='*', type=str)

    args = parser.parse_args()
    # make the job uppercase
    args.job[0] = args.job[0].upper()
    if len(args.job) == 1:
        args.job.append('')

    if args.job[0] != 'STATUS':
        housekeeper.info('Start program')

    # prepare asyncio event loop
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(exception_handler)

    # create async serial connection
    protocol = MecmastProtocol(0x33, 63, args.job, loop)
    coro = serial_asyncio.create_serial_connection(loop, lambda: protocol, '/dev/mecmast', baudrate=9600)
    coro_with_timeout = asyncio.wait_for(coro, 5)
    try:
        loop.run_until_complete(coro_with_timeout)
    except asyncio.TimeoutError:
        housekeeper.critical('Connection to serial port /dev/mecmast not possible!')
        return

    issue_final_stop = False
    # schedule a watchdog-like timeout. If something goes wrong, terminate the program after 120 seconds
    timeout = 120
    loop.create_task(asyncio.wait_for(asyncio.sleep(timeout + 1), timeout))

    try:
        # this is actually the "main loop"
        loop.run_forever()
    except (KeyboardInterrupt, SystemExitStop, TimeoutError):
        issue_final_stop = True
    except SystemExit:
        pass
    finally:
        # cleanly exit
        if args.job[0] != 'STATUS':
            housekeeper.info('Terminate program.')
        protocol.transport.close()
        if issue_final_stop:
            loop.run_until_complete(stop())


# ******************************
if __name__ == "__main__":
    main(sys.argv[1:])