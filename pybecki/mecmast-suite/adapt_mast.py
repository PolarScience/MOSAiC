#!/home/venv/bin/python

import asyncio
import sys
import pygl
import argparse
import json
from statistics import mean, StatisticsError

housekeeper = pygl.get_housekeeper(group='AdaptiveMast')


async def client_connected(reader, writer):
    """
    This coroutine reads the distance values from the SHM50 local server until it is cancelled

    :param reader: StreamReader instance for reading from the socket
    :param writer: StreamWriter instance for writing to the socket
    :return: a time series of the spatially averaged distance values
    :rtype: list
    """
    time_series = list()

    try:
        while True:
            line = await reader.readline()
            housekeeper.debug('Got data line: {}'.format(line))

            shm50_dict = json.loads(line.decode())

            if shm50_dict['SNOW']:
                # only take distance measurements, when they fulfill minimal quality standards
                depths = [depth for depth, error, intensity in
                          zip(shm50_dict['DEPTH'], shm50_dict['ERROR'], shm50_dict['INTENSITY'])
                          if error == '00' and intensity > 0]
                try:
                    spatial_mean = mean(depths)
                except StatisticsError:
                    housekeeper.warning('No valid distance data from all 3 sensors')
                else:
                    housekeeper.debug('Spatial mean: {:.4} m'.format(spatial_mean))
                    time_series.append(spatial_mean)
    except asyncio.CancelledError:
        pass

    writer.close()

    return time_series


def main(argv):
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Control and the Fireco Mecmast.')
    parser.add_argument('--loglevel', help="the logging level", type=str, default="WARNING")
    parser.add_argument('--avg_time', help="averaging time in minutes", type=int, default=5)
    parser.add_argument('--threshold', help="absolute distance in mm, above which the mast is adapted. Default is 50",
                        type=int, default=50)
    parser.add_argument('--disable_slope', help="disable the distance measurement correction due to the slope angle",
                        action='store_true')
    parser.add_argument('host', help="ip address where the distance sensor is serving", type=str)
    parser.add_argument('port', help="tcp port where the distance sensor is serving", type=int)

    args = parser.parse_args()

    # prepare the asyncio event loop
    loop = asyncio.get_event_loop()

    # create a connection to the SHM50 local server socket
    coro = asyncio.open_connection(args.host, args.port, loop=loop)
    (reader, writer) = loop.run_until_complete(coro)

    # prepare the SHM50 reading task
    read_task = loop.create_task(client_connected(reader, writer))
    try:
        # run the SHM50 reading task until the time specified by avg_time has passed
        time_series = loop.run_until_complete(asyncio.wait_for(read_task, args.avg_time*60-1))
    except asyncio.TimeoutError:
        time_series = read_task.result()
    except (KeyboardInterrupt, OSError):
        return

    # handle the distance time series
    try:
        mean_distance = mean(time_series)
    except StatisticsError:
        housekeeper.warning('No distance data or snow present. Will not move mast')
    else:
        housekeeper.info('Average distance over the last {} minutes was {:.4} m'.format(args.avg_time, mean_distance))
        if not args.disable_slope:
            mean_distance = mean_distance / 0.845
            housekeeper.info('True distance (slope corrected) is therefore {:.4} m'.format(mean_distance))
        mean_distance_mm = mean_distance*1000
        subprocess_args = ['auto']
        if args.threshold < mean_distance_mm:
            subprocess_args.append('+{}'.format(int(mean_distance_mm)))
        elif -args.threshold > mean_distance_mm:
            subprocess_args.append('{}'.format(int(mean_distance_mm)))
        if len(subprocess_args) > 1:
            housekeeper.info('Will try to move mast {} mm'.format(subprocess_args[1]))

            coro = asyncio.create_subprocess_exec('mecmast.py', *subprocess_args,
                                                  stdout=asyncio.subprocess.DEVNULL,
                                                  stderr=asyncio.subprocess.DEVNULL)
            process = loop.run_until_complete(coro)

            loop.run_until_complete(process.wait())
            if process.returncode == 0:
                housekeeper.info('mecmast.py executed successfully')
        else:
            housekeeper.info("Will not move mast")


# ******************************
if __name__ == "__main__":
    main(sys.argv[1:])