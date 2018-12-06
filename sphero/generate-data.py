"""
"""

import os, sys, asyncio, time
import numpy as np
import pandas as pd
import matplotlib as plt
from tabulate import tabulate
import spheropy
from bluetooth_interface import BluetoothInterface

NUM_DATA_POINTS = 100
MAX_SPEED = 255
MAX_HEADING = 359
MAX_DURATION = 1.0
MIN_DURATION = 0.01

async def main():
    df = create_and_initialize_dataframe()

    # setup the sphero
    sphero = create_and_connect_sphero()

    # configure collision detection
    collision_occured = False
    col_x = 0
    col_y = 0
    def handle_collision(collision_data):
        nonlocal collision_occured
        nonlocal col_x
        nonlocal col_y
        collision_occured = True
        col_x = collision_data.x_impact
        col_y = collision_data.y_impact

    sphero.on_collision.append(handle_collision)
    await sphero.configure_collision_detection(
        turn_on_collision_detection=True,
        x_t=30,
        x_speed=50,
        y_t=30,
        y_speed=50,
        collision_dead_time=10)

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
        time.sleep(df.loc[index, 'desired_duration'])

        # TODO: might need to calculate duration based on the time
        # right before we send the next roll command
        locator_info1 = await sphero.get_locator_info()
        df.loc[index, 'x0'] = locator_info0.pos_x
        df.loc[index, 'y0'] = locator_info0.pos_y
        df.loc[index, 'vel_x0'] = locator_info0.vel_x
        df.loc[index, 'vel_y0'] = locator_info0.vel_y
        df.loc[index, 'x1'] = locator_info1.pos_x
        df.loc[index, 'y1'] = locator_info1.pos_y
        df.loc[index, 'vel_x1'] = locator_info1.vel_x
        df.loc[index, 'vel_y1'] = locator_info1.vel_y
        df.loc[index, 'col_x'] = col_x
        df.loc[index, 'col_y'] = col_y
        # if collision_occured:
        #     print("Collision occured during data point: {}!".format(index))
        #     # Drop this index to the end
        #     df.drop(df.index[index + 1:], inplace=True)
        #     break
        df.loc[index, 'actual_duration'] = time.time() - time0

    # stop the sphero when done
    print('Stoping the Sphero.')
    await sphero.roll(speed=0, heading_in_degrees=0)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    save_dataframe(df)


def create_and_initialize_dataframe():
    df = pd.DataFrame(index=range(NUM_DATA_POINTS),
        columns=['speed', 'heading', 'desired_duration', # control inputs
        'actual_duration', 'x0', 'y0', 'vel_x0', 'vel_y0', 'col_x', 'col_y',
        'x1', 'y1', 'vel_x1', 'vel_y1'] # lables
    )

    # Create random values for control inputs
    df.loc[:, 'speed'] = np.random.randint(low=0, high=(MAX_SPEED + 1), size=NUM_DATA_POINTS)
    df.loc[:, 'heading'] = np.random.randint(low=0, high=(MAX_HEADING + 1), size=NUM_DATA_POINTS)
    df.loc[:, 'desired_duration'] = np.random.random_sample(size=NUM_DATA_POINTS)*(MAX_DURATION - MIN_DURATION) + MIN_DURATION
    return df

def create_and_connect_sphero():
    MAX_RETRY_COUNT = 3
    socket = BluetoothInterface()
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
    return spheropy.Sphero(socket)

def save_dataframe(df):
    print('Saving data to csv file')
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(csv_dir, exist_ok=True)
    csv_file_name = 'sphero-data-{}.csv'.format(time.strftime("%Y%m%d-%H%M%S"))
    csv_file_path = os.path.join(csv_dir, csv_file_name)
    df.to_csv(csv_file_path)

if __name__ == '__main__':
    main_loop = asyncio.get_event_loop()
    main_loop.run_until_complete(main())