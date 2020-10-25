#!/home/venv/bin/python

import sys
import argparse
from pymodbus.client.sync import ModbusTcpClient
import logging
import json

logger = logging.getLogger('ioLogik')

iologik_types = {1214: {"di_address": 48, "do_address": 32, "nr_di": 6, "nr_do": 6}}


class IoLogik(ModbusTcpClient):
    types = {1214: {"input_address": 48, "output_address": 32, "nr_di": 6, "nr_do": 6}}

    def __init__(self, dev_type=1214, *args, **kwargs):
        """
        :param dev_type: device type out of keys in iologik_types
        """
        super().__init__(*args, **kwargs)
        self.type = dev_type
        self.input_addr = self.types[dev_type]["input_address"]
        self.output_addr = self.types[dev_type]["output_address"]

    def read_inputs(self):
        """
        Reads the input register

        :return: integer value of the input register
        """
        return self.read_input_registers(self.input_addr)

    def read_outputs(self):
        """
        Reads the output register

        :return: integer value of the input register
        """
        return self.read_holding_registers(self.output_addr)

    def write_outputs(self, register_value):
        """
        Writes the desired value to the output register

        :param register_value: integer to be written to the output register
        """
        self.write_register(self.output_addr, register_value)


def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser(description='MOXA ioLogik TCP Modbus helper')
    parser.add_argument('ip', help="IP of the ioLogik", type=str)
    set_coil = parser.add_mutually_exclusive_group()
    set_coil.add_argument('-w', '--write', help="value to write to the register for all DO's", type=str, nargs=1)
    set_coil.add_argument('-s', '--set_coil', help="sets the coil at the given address to true", type=int, nargs=1,
                          default=None)
    set_coil.add_argument('-u', '--unset_coil', help="unsets the coil at the given address to false", type=int, nargs=1,
                          default=None)
    set_coil.add_argument('--read_input_registers', help="read an input register, provide starting address and count",
                          type=int, nargs=2)
    set_coil.add_argument('--read_coils', help="read coils, provice starting address and count", type=int, nargs=2)
    parser.add_argument('--loglevel', help="the logging level", type=str, default="WARNING", nargs=1)
    parser.add_argument('--iologik_type', help="ioLogik type, number only", type=int, default=1214, nargs=1)

    args = parser.parse_args(argv)

    # parse the value to write, if given
    register_value = None
    try:
        register_value = int(args.write[0], 0)    # when specifying base 0, python guesses the base (0x23, 0b10011, etc)
    except ValueError:
        raise argparse.ArgumentError('Value given for -w is not a number.')
    except AttributeError:
        pass
    except TypeError:
        pass

    # initialize the modbus client
    iologik = IoLogik(host=args.ip, type=args.iologik_type)

    # load some defaults from the ioLogik types table
    nr_di = iologik.types[iologik.type]["nr_di"]
    nr_do = iologik.types[iologik.type]["nr_do"]

    # initialize the logger
    logging.basicConfig(level=args.loglevel)

    # if option --read_input_registers is chosen, do only this and return
    if args.read_input_registers:
        # read the input registers from the iologik
        result = iologik.read_input_registers(args.read_input_registers[0], args.read_input_registers[1])

        # prepare a dictionary for the JSON serialized output. Key=Address, Value=register value
        outdict = dict()
        for i in range(args.read_input_registers[1]):
            outdict.update({args.read_input_registers[0] + i: result.getRegister(i)})
        outstr = json.dumps(outdict)
        logger.info(outstr)
        print(outstr)
        return

    if args.read_coils:
        # read the input coil from the iologik
        result = iologik.read_coils(args.read_coils[0], args.read_coils[1])

        # prepare a dictionary for the JSON serialized output. Key=Starting Address, Value=list of register values
        outdict = dict()
        for i in range(args.read_coils[1]):
            outdict.update({args.read_coils[0] + i: result.getBit(i)})
        outstr = json.dumps(outdict)
        logger.info(outstr)
        print(outstr)
        return

    # read the registers and write, if necessary
    try:
        if register_value is not None:
            iologik.write_outputs(register_value)
        elif args.set_coil is not None:
            iologik.write_coil(args.set_coil[0], True)
        elif args.unset_coil is not None:
            iologik.write_coil(args.unset_coil[0], False)
        di = iologik.read_inputs()
        do = iologik.read_outputs()
    except ConnectionError:
        logger.error('Could not connect to ioLogik at ' + args.ip)
        raise
    finally:
        iologik.close()

    # print the output to stdout in JSON format
    formstr_di = "{:0>" + str(nr_di) + "b}"
    formstr_do = "{:0>" + str(nr_do) + "b}"
    outdict = {'DI': formstr_di.format(di.getRegister(0)),
               'DO': formstr_do.format(do.getRegister(0))}
    outstr = json.dumps(outdict)
    logger.info(outstr)
    print(outstr)


# ******************************
if __name__ == "__main__":
    main(sys.argv[1:])
