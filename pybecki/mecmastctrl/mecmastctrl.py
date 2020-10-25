#!/home/venv/bin/python3

"""Copy of  control the MecMast.

Created: 22.09.17 (PM and IB)
This is the first version that managed to move the mast at AWS
Changes:
23.10.17 (IB): Included input arguments with argparse
28.08.18 (IB): Tested this version of the script and it seems to work. It moves up, down with move_to. Also it shows
the status permanently on moving until it has reached the input height, where the status jumps to wait command.
THe script uses serialio.py.

"""

import sys
import argparse

""" Append the path of the library to the system path """
sys.path.append('/home/awsdaq/pythontests/test_mecmast/')


from serialio import Serial
import time
import pdb


class MecmastCtrl:
    """Presets for different mast-move commands """
    """3rd byte: Defines the MOT1 command (move to, move up/down, stop etc.
    4-6th bytes: Define the position where the mast should go in automatic mode. (standard (0cm): x00,x00,x04, 20cm: x00,x00,x63)
    DONT FORGET TO ADJUST CHECKSUM (second last byte, must be sum of bytes 2-9)!"""
    reset = b'\x02\x33\x55\x00\x00\x00\x55\x00\x00\x00\xaa\x04'
    unblock = b'\x02\x33\x06\x00\x00\x00\x00\x00\x00\x00\x06\x04'
    moveup = b'\x02\x33\x01\x00\x08\x00\x00\x00\x00\x00\x09\x04'
    stop = b'\x02\x33\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04'
    #moveto = b'\x02\x33\x03\x00\x00\x04\x00\x00\x00\x00\x07\x04' #not needed anymore, now: mmc.cmdstr

    keyup = b'\x02\x55\x31\x00\x04'

    def __init__(self, *args, **kwargs):
        """Attribute"""
        self.cycle = [(0, 'wait_release'), (1, 'wait_command'), (2, 'manual_move_down'), (3, 'manual_move_up'),
                      (4, 'auto_move'), (5, 'set_min_height'), (6, 'set_max_height'), (7, 'set_default'), (8, 'unkown'),
                      (9, 'error'), (10, 'initialize')]
        self.lift_motor = {'up_cnt': 0, 'down_cnt': 0, 'work_hrs': 0, 'current': 0, 'range': [1430, 4021], 'encoder': 0,
                           'cycle': self.cycle[10], 'error_undercurrent': 0, 'error_overcurrent': 0, 'error_encoder': 0}
        self.rot_motor = {'up_cnt': 0, 'down_cnt': 0, 'work_hrs': 0, 'current': 0, 'range': [0, 360], 'encoder': 0,
                          'cycle': self.cycle[10], 'error_undercurrent': 0, 'error_overcurrent': 0, 'error_encoder': 0}
        self.keyboard = {'poll': 0, 'echo': 0, 'value': 0}
        self.serial = {'baudrate': kwargs.get('baudrate', 9600), 'port': kwargs.get('port', '/dev/mecmast'),
                       'timeout': kwargs.get('timeout', 1)}
        self.status = {'firmware': float(0), 'output': [0, 0, 0], 'input': [0, 0, 0, 0], 'level': 0, 'crc_calc': 0,
                       'crc_read': 0, 'error_connection': 0, 'error_short_circuit': 0, 'cycle': self.cycle[10],
                       'heartbeat': 0}
        self.sio = Serial(self.serial['port'], self.serial['baudrate'], timeout=self.serial['timeout'])
        self.cmdstr = bytearray(12)  # bytearray for the command to the mast

    """Methoden (Funktion)"""

    def prep_cmd(self, lift_cmd='', height_mm=0, outputs=[0, 0, 0], mm_per_digit=2):
        # def send_cmd(self, lift_cmd='', height_mm=0, rot_cmd='', angle_deg=0, outputs=[0, 0, 0], mm_per_digit=2):
        """ Calculate ENCODER and POTENTIOMETER value """
        encval = height_mm / mm_per_digit
        # encval = (height_mm - self.lift_motor['range'][0]) * 8191 / (self.lift_motor['range'][1]-self.lift_motor['range'][0])
        encval = int(min(1281, max(0, round(encval, 0))))
        # potval = (angle_deg - self.rot_motor['range'][0]) * 1023 / (self.rot_motor['range'][1]-self.rot_motor['range'][0])
        # potval = int(min(1023,max(0,round(potval,0))))

        #print('ENCVAL = %s' % encval)
        #print('ENCVAL IN BYTES' , encval.to_bytes(3, byteorder='big'))

        #print('POTVAL = %s' % potval)

        """ Set STX to 0x02 and ADRESS to 0x55 """
        self.cmdstr[0:2] = b'\x02\x33'
        #print('FIRST TWO BYTES COMMAND:', self.cmdstr)

        """ Convert LIFT_MOTOR_CMD (set mot1 command and position) """
        if lift_cmd.upper() == 'STOP':
            self.cmdstr[2:6] = b'\x00\x00\x00\x00'
        elif lift_cmd.upper() == 'MOVE_UP':
            self.cmdstr[2:6] = b'\x01\x00\x00\x00'
        elif lift_cmd.upper() == 'MOVE_DN':
            self.cmdstr[2:6] = b'\x02\x00\x00\x00'
        elif lift_cmd.upper() == 'MOVE_TO':
            self.cmdstr[2:6] = b'\x03' + encval.to_bytes(3, byteorder='big')
        elif lift_cmd.upper() == 'SET_MIN':
            self.cmdstr[2:6] = b'\x04' + encval.to_bytes(3, byteorder='big')
        elif lift_cmd.upper() == 'SET_MAX':
            self.cmdstr[2:6] = b'\x05' + encval.to_bytes(3, byteorder='big')
        elif lift_cmd.upper() == 'UNBLOCK':
            self.cmdstr[2:6] = b'\x06\x00\x00\x00'
        elif lift_cmd.upper() == 'RESET':
            self.cmdstr[2:6] = b'\x55\x00\x00\x00'
            self.cmdstr[6:10] = b'\x55\x00\x00\x00'
        elif lift_cmd.upper() == 'ALARM':
            self.cmdstr[2:6] = b'\x5A\x00\x00\x00'
            self.cmdstr[6:10] = b'\x5A\x00\x00\x00'
        elif lift_cmd.upper() == '':
            pass
        else:
            raise RuntimeError(
                'Invalid LIFT_MOTOR_COMMAND! Use: STOP, MOVE_UP, MOVE_DN, MOVE_TO, SET_MIN, SET_MAX, UNBLOCK, RESET or ALARM')

        # """ Convert ROT_MOTOR_CMD """
        # if rot_cmd.upper() == 'STOP':
        #    self.cmdstr[6:9] = b'\x00\x00\x00'
        # elif rot_cmd.upper() == 'MOVE_UP':
        #    self.cmdstr[6:9] = b'\x01\x00\x00'
        # elif rot_cmd.upper() == 'MOVE_DN':
        #    self.cmdstr[6:9] = b'\x02\x00\x00'
        # elif rot_cmd.upper() == 'MOVE_TO':
        #    self.cmdstr[6:9] = b'\x03' + potval.to_bytes(2,byteorder='big')
        # elif rot_cmd.upper() == 'SET_MIN':
        #    self.cmdstr[6:9] = b'\x04' + potval.to_bytes(2,byteorder='big')
        # elif rot_cmd.upper() == 'SET_MAX':
        #    self.cmdstr[6:9] = b'\x05' + potval.to_bytes(2,byteorder='big')
        # elif lift_cmd.upper() == 'UNBLOCK':
        #    self.cmdstr[6:9] = b'\x06\x00\x00'
        # elif rot_cmd.upper() == 'RESET':
        #    self.cmdstr[2:6] = b'\x55\x00\x00'
        #    self.cmdstr[6:9] = b'\x55\x00\x00'
        # elif rot_cmd.upper() == 'ALARM':
        #    self.cmdstr[2:6] = b'\x5A\x00\x00'
        #    self.cmdstr[6:9] = b'\x5A\x00\x00'
        # elif rot_cmd.upper() == '':
        #    pass
        # else:
        #    raise RuntimeError('Invalid ROT_MOTOR_COMMAND! Use: STOP, MOVE_UP, MOVE_DN, MOVE_TO, SET_MIN, SET_MAX, UNBLOCK, RESET or ALARM')
        #
        """ Set OUTPUTS (Not documented and not used as far as I know)"""
        outval = 0
        if outputs[0]:
            outval = outval | 1
        if outputs[1]:
            outval = outval | 2
        if outputs[2]:
            outval = outval | 4

        outval = int(outval)
        #print('OUTPUT STATUS: = %s' % outval)
        # pdb.set_trace()
        self.cmdstr[9:10] = outval.to_bytes(1, byteorder='big')


        """ Calculate Checksum """
        self.cmdstr[10:11] = (sum(self.cmdstr[2:10]) % 256).to_bytes(1, byteorder='big')


        """ Set EOT """
        self.cmdstr[11:12] = b'\x04'
        #print('Command string: ' + self.cmdstr.hex())

    def read33(self):
        #print('Start read33, read message from Mast to PC')
        """ Read Messages sent from the Mast to the PC (33 is adress of PC) """
        """Byte sequence
      0: Start byte 0x02
      1: Address: PC = 0x33
      2,3: Software version, higher and lower byte
      4-6: No of starts of lifting motor (MOT1) upwards
      7-9: No of starts of MOT1 downwards
      10-12: Working time of MOT1
      13,14: Current on MOT1
      15-17: No of starts of rotating motor upwards (MOT2)
      18-20: No of starts of MOT2 downwards
      21-23: Working time MOT2
      24,25: Current MOT2
      26: Status of exits
      27: Status of entrances
      28,29: Angle of MOT2 (0-1023)
      30-32: Height of MOT1 (0-8191)
      33: Level of radio Signal
      34: Alarms: 
      35: Status MOT1
      36: Status MOT2
      37: Cycle status
      38: Checksum
      39: End byte 0x04
      """
        nbytes, msg = self.sio.readmsg(startseq=b'\x04\x02\x33', bytestoread=40)  # look for start sequence 02 33
        #print('Read %03i chars: %s' % (nbytes, msg.hex()))

        self.status['crc_calc'] = sum(msg[3:39]) % 256  # calculate checksum
        self.status['crc_read'] = msg[39]  # read checksum, must be same as crc_calc
        if self.status['crc_read'] != self.status['crc_calc']:
            print('ERROR: Bad Message, checksum does not coincide')
            return

        self.status['firmware'] = float(msg[3]) + 0.01 * float(msg[4])
        self.status['output'] = [msg[27] & 1 != 0, msg[27] & 2 != 0, msg[27] & 4 != 0]
        self.status['heartbeat'] = msg[27] & 248 != 0 #?? What is this??
        self.status['input'] = [msg[28] & 1 != 0, msg[28] & 2 != 0, msg[28] & 4 != 0, msg[28] & 8 != 0]
        self.status['level'] = msg[34]  # level of radio signal
        self.status['error_short_circuit'] = msg[35] & 16 != 0
        self.status['error_connection'] = msg[35] & 128 != 0
        try:  # TODO diesen Status (Cycle Status) anders aufschlüsseln (nur werte 7 od 9 gültig)
            self.status['cycle'] = self.cycle[msg[37]]
        except:
            self.status['cycle'] = (msg[37], 'unknown ' + str(msg[37]))

        self.lift_motor['up_cnt'] = int.from_bytes(msg[5:8], byteorder='big', signed=False)
        self.lift_motor['down_cnt'] = int.from_bytes(msg[8:11], byteorder='big', signed=False)
        self.lift_motor['work_hrs'] = int.from_bytes(msg[11:14], byteorder='big', signed=False)
        self.lift_motor['current'] = 0.1 * int.from_bytes(msg[14:16], byteorder='big', signed=False)
        self.lift_motor['encoder'] = int.from_bytes(msg[31:34], byteorder='big', signed=False)  # Height of the mast
        try:
            self.lift_motor['cycle'] = self.cycle[msg[36]]
        except:
            self.lift_motor['cycle'] = (msg[36], 'unkown ' + str(msg[36]))
        self.lift_motor['error_undercurrent'] = msg[35] & 1 != 0
        self.lift_motor['error_overcurrent'] = msg[35] & 2 != 0
        self.lift_motor['error_encoder'] = msg[35] & 32 != 0

        self.rot_motor['up_cnt'] = int.from_bytes(msg[16:19], byteorder='big', signed=False)
        self.rot_motor['down_cnt'] = int.from_bytes(msg[19:22], byteorder='big', signed=False)
        self.rot_motor['work_hrs'] = int.from_bytes(msg[22:25], byteorder='big', signed=False)
        self.rot_motor['current'] = 0.1 * int.from_bytes(msg[25:27], byteorder='big', signed=False)
        self.rot_motor['encoder'] = int.from_bytes(msg[29:31], byteorder='big', signed=False)
        try:
            self.rot_motor['cycle'] = self.cycle[msg[38]]
        except:
            self.rot_motor['cycle'] = (msg[38], 'unkown ' + str(msg[38]))
        self.rot_motor['error_undercurrent'] = msg[35] & 4 != 0
        self.rot_motor['error_overcurrent'] = msg[35] & 8 != 0
        self.rot_motor['error_encoder'] = msg[35] & 64 != 0

    def read77(self):
        """Read sent to the UM451 board """
        nbytes, msg = self.sio.readmsg(startseq=b'\x04\x02\x77', bytestoread=37)
        print('Read %03i chars: %s' % (nbytes, msg.hex()))

        self.status['crc_calc'] = sum(msg[3:36]) % 256
        self.status['crc_read'] = msg[36]
        if self.status['crc_read'] != self.status['crc_calc']:
            print('ERROR: Bad Message')
            return

        self.status['firmware'] = float(msg[3]) + 0.01 * float(msg[4])
        self.status['output'] = [msg[27] & 1 != 0, msg[27] & 2 != 0, msg[27] & 4 != 0]
        self.status['heartbeat'] = msg[27] & 248 != 0
        self.status['input'] = [msg[28] & 1 != 0, msg[28] & 2 != 0, msg[28] & 4 != 0, msg[28] & 8 != 0]
        self.status['level'] = msg[34]
        self.status['error_short_circuit'] = msg[35] & 16 != 0
        self.status['error_connection'] = msg[35] & 128 != 0

        self.lift_motor['up_cnt'] = int.from_bytes(msg[5:8], byteorder='big', signed=False)
        self.lift_motor['down_cnt'] = int.from_bytes(msg[8:11], byteorder='big', signed=False)
        self.lift_motor['work_hrs'] = int.from_bytes(msg[11:14], byteorder='big', signed=False)
        self.lift_motor['current'] = 0.1 * int.from_bytes(msg[14:16], byteorder='big', signed=False)
        self.lift_motor['encoder'] = int.from_bytes(msg[31:34], byteorder='big', signed=False)
        self.lift_motor['error_undercurrent'] = msg[35] & 1 != 0
        self.rot_motor['error_overcurrent'] = msg[35] & 8 != 0
        self.rot_motor['error_encoder'] = msg[35] & 64 != 0

        self.rot_motor['up_cnt'] = int.from_bytes(msg[16:19], byteorder='big', signed=False)
        self.rot_motor['down_cnt'] = int.from_bytes(msg[19:22], byteorder='big', signed=False)
        self.rot_motor['work_hrs'] = int.from_bytes(msg[22:25], byteorder='big', signed=False)
        self.rot_motor['current'] = 0.1 * int.from_bytes(msg[25:27], byteorder='big', signed=False)
        self.rot_motor['encoder'] = int.from_bytes(msg[29:31], byteorder='big', signed=False)
        self.rot_motor['error_undercurrent'] = msg[35] & 4 != 0
        self.rot_motor['error_overcurrent'] = msg[35] & 8 != 0
        self.rot_motor['error_encoder'] = msg[35] & 64 != 0

    def read55(self):
        nbytes, msg = self.sio.readmsg(startseq=b'\x04\x02\x55', bytestoread=5)
        print('Read %03i chars: %s' % (nbytes, msg.hex()))

        """Read sent to the keyboard """
        self.keyboard['poll'] = msg[3] == ord('\x31')
        self.keyboard['echo'] = msg[3] == ord('\x30')
        self.keyboard['value'] = msg[4]


if __name__ == '__main__':
    """Instanz"""

    mmc = MecmastCtrl()  # refers to the class MecmastCtrl that is defined above
    dt = 0.4    #for sleeping time

    # -------------------------------------------------
    # Read the Argparse inputs and decide what to do:
    parser = argparse.ArgumentParser(description='Input: -m N for movement to N mm, -s for status request or -g to only get the height of the mast')
    parser.add_argument("-m", "--moveto", type= int, choices =range(5,2560), metavar = '[5-2560]', help="type in -m followed by the height in mm (range: 8...2560mm)")
   # parser.add_argument("-u", "--up", help="type in -u followed by amount of up movement in mm", type=int)
   # parser.add_argument("-d", "--down", help="type in -d followed by amount of down movement in mm", type=int)
    parser.add_argument("-s", "--status", action = 'store_true')
    parser.add_argument("-g", "--getheight", action = 'store_true')
    parser.add_argument("-r", "--reset", action = 'store_true')



    args = parser.parse_args()
    if args.moveto:
        input_height = args.moveto
        print('Mecmast_ctrl: Move to mast height (mm):', input_height)
        mmc.prep_cmd(lift_cmd="MOVE_TO", height_mm=input_height) #send new input height to set up the new move_to string

        # Prepare mast
        time.sleep(0.010)  # wait command before writing to mast
        mmc.sio.write(mmc.stop)
        print('Write STOP: ' + mmc.stop.hex())
        time.sleep(dt)
        mmc.read33()
        time.sleep(0.01)
        mmc.sio.write(mmc.cmdstr)
        # print('Status: ', mmc.status) #For debugging
        # print('Lift-motor: ', mmc.lift_motor) #For debugging
        # print('Write move_to command: ' + mmc.cmdstr.hex())  #For debugging

        time.sleep(dt)

        while True:
            mmc.read33()
            #print('Status: ', mmc.status)
            #print('Lift-motor: ', mmc.lift_motor)
            print('Moving...')
            time.sleep(0.010)
            if mmc.lift_motor['cycle'] == mmc.cycle[0]:  # has stopped, waiting for release (stop command)
                mmc.sio.write(mmc.stop)
                #print('Writing stop: ' + mmc.stop.hex())
                print('Stopped')
            elif mmc.lift_motor['cycle'] == mmc.cycle[4]:  # is moving
                mmc.sio.write(mmc.cmdstr)
               # print('writing: ' + mmc.cmdstr.hex())
            elif mmc.lift_motor['cycle'] == mmc.cycle[1]:  # released (stop command received)
                print('cycle 1: Wait command')
                break
            time.sleep(dt)
    elif args.status:
        mast_status = args.status
        mmc.read33()
        height_m=mmc.lift_motor['encoder']*0.002
        print('STATUS OF MAST:')
        print('MAST_LEVEL [m]: ', height_m)
        print('Status: ', mmc.status)
        print('Lift-motor: ', mmc.lift_motor)


    elif args.getheight:
        mast_height = args.getheight
        mmc.read33()
        height_m = mmc.lift_motor['encoder'] * 0.002
        print(height_m)

    elif args.reset:
        """    reset = b'\x02\x33\x55\x00\x00\x00\x55\x00\x00\x00\xaa\x04' """
        mast_reset = args.reset
        mmc.read33()

        mmc.prep_cmd(lift_cmd="RESET") #send reset command
    #TODO: prep cmd reset vorbereiten, existiert noch nicht.
        # Prepare mast
        time.sleep(0.010)  # wait command before writing to mast
        mmc.sio.write(mmc.stop)
        print('Write STOP: ' + mmc.stop.hex())
        time.sleep(dt)
        mmc.read33()
        time.sleep(0.01)



    else:
        print('You did not choose -m (moveto), -s (status request) nor -g (height request). Choose one.')
    #send height for calibration

    #---------------------------------------------------
    #Send commands to mast in right order

    #read mast Infos and print
    """mmc.read33()  # read from address 0x33 (mast)
    print('STATUS: ', mmc.status)
    print('LIFT-MOTOR: ', mmc.lift_motor)
    print('MAST_LEVEL: ', mmc.lift_motor['encoder'])
    """




        # pdb.set_trace()
