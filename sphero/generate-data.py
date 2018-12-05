"""
"""

import os, sys, asyncio, time
import numpy as np
import pandas as pd
import matplotlib as plt
from tabulate import tabulate
import spheropy
from bluetooth_interface import BluetoothInterface

NUM_DATA_POINTS = 10

async def main():
    #
    # [speed, heading, duration, x0, y0, vel_x0, vel_y0, x1, y1, vel_x1, vel_y1, actual_duration]
    # TODO: delete data if not needed
    data = np.empty((NUM_DATA_POINTS, 12))
    data[:, 0] = np.random.randint(low=0, high=256, size=NUM_DATA_POINTS)
    data[:, 1] = np.random.randint(low=0, high=360, size=NUM_DATA_POINTS)
    data[:, 2] = np.random.random_sample(size=NUM_DATA_POINTS)*2 + 0.01
    df = pd.DataFrame(index=range(NUM_DATA_POINTS), columns=['speed', 'heading', 'duration', 'x0', 'y0', 'vel_x0', 'vel_y0', 'x1', 'y1', 'vel_x1', 'vel_y1', 'actual_duration'])
    df.loc[:, 'speed'] = np.random.randint(low=0, high=256, size=NUM_DATA_POINTS)
    df.loc[:, 'heading'] = np.random.randint(low=0, high=256, size=NUM_DATA_POINTS)
    df.loc[:, 'duration'] = np.random.random_sample(size=NUM_DATA_POINTS)*2 + 0.01

    # setup the sphero
    socket = BluetoothInterface()
    MAX_RETRY_COUNT = 3
    socket_connected = False
    for try_count in range(MAX_RETRY_COUNT):
        try:
            socket.connect()
            socket_connected = True
            break
        except OSError:
            print('Failed to connect {}.'.format(try_count))

    if not socket_connected:
        print('Could not connect to Sphero.')
        sys.exit(1)
    sphero = spheropy.Sphero(socket)

    for index in range(NUM_DATA_POINTS):
        locator_info0 = await sphero.get_locator_info()

        heading = df.loc[index, 'heading']
        speed = df.loc[index, 'speed']
        print('Executing roll with heading: {} and speed: {}'.format(heading, speed))
        time0 = time.time()
        await sphero.roll(
            speed=speed,
            heading_in_degrees=heading,
            wait_for_response=False, # try not waiting for the response
            response_timeout_in_seconds=0.2)
        time.sleep(df.loc[index, 'duration'])
        time1 = time.time()

        # TODO: might need to calculate duration based on the time
        # right before we send the next roll command
        actual_duration = time1 - time0
        locator_info1 = await sphero.get_locator_info()
        df.loc[index, 'x0'] = locator_info0.pos_x
        df.loc[index, 'y0'] = locator_info0.pos_y
        df.loc[index, 'vel_x0'] = locator_info0.vel_x
        df.loc[index, 'vel_y0'] = locator_info0.vel_y
        df.loc[index, 'x1'] = locator_info1.pos_x
        df.loc[index, 'y1'] = locator_info1.pos_y
        df.loc[index, 'vel_x1'] = locator_info1.vel_x
        df.loc[index, 'vel_y1'] = locator_info1.vel_y
        df.loc[index, 'actual_duration'] = actual_duration

    # stop the sphero when done
    print('Stoping the Sphero')
    await sphero.roll(speed=0, heading_in_degrees=0)
    print(tabulate(df, headers='keys', tablefmt='psql'))

if __name__ == '__main__':
    main_loop = asyncio.get_event_loop()
    main_loop.run_until_complete(main())