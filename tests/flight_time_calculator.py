import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Battery:
    def __init__(self, cells, capacity, mass):
        self.cells = cells
        self.capacity = capacity
        self.mass = mass

    def __repr__(self):
        return 'Battery(cells={}, capacity={}, mass={})'.format

def read_motor_test_report(path):
    '''
    # Tested only with data from T-Motors website
    :param path: path to the motor test report
    :return: dataframe of motor test report
    '''
    # read motor test report with pandas, no header, no index
    motor_test_report = pd.read_csv(path, header=None, index_col=False)
    # add headers to the dataframe
    motor_test_report.columns = ['Type', 'Propeller', 'Throttle', 'Thrust', 'Voltage', 'Current', 'RPM', 'Power', 'Efficiency', 'Temperture']
    # convert the 'Thrust' column to float
    motor_test_report['Throttle'] = motor_test_report['Throttle'].str.replace('%', '').astype(float)

    motor_test_report['Power'] = motor_test_report['Power'].str.replace(',', '.').astype(float)
    # find in which row 'thrust' is 100%
    motor_test_report_list = []
    idx = np.append(-1, np.append(motor_test_report[motor_test_report['Throttle'] == 100].index.values, len(motor_test_report)))
    # slice the dataframe where 'thrust' is 100%
    for b, n in zip(idx[:-1], idx[1:]):
        motor_test_report_list.append(motor_test_report[b+1:n+1])
    if len(motor_test_report_list[-1]) == 0:
        del motor_test_report_list[-1]
    return motor_test_report_list


def model_motor_test_report(motor_test_report, degree=3):
    '''
    :param motor_test_report: dataframe of motor test report
    :param degree: degree of the polynomial
    :return: polynomial model of power as function of thrust
    '''
    thrust = motor_test_report['Thrust'].values
    power = motor_test_report['Power'].values
    # add 0 thrust and 0 power to the corresponding arrays
    thrust = np.append(0.0, thrust)
    power = np.append(0.0, power)
    return np.poly1d(np.polyfit(thrust, power, degree))


def throttle_and_current_from_thrust(thrust_value_at_hover, motor_test_report, degree=3):
    '''
    :param motor_test_report: dataframe of motor test report
    :param degree: degree of the polynomial
    :return: polynomial model of throttle and current as function of thrust
    '''
    thrust = motor_test_report['Thrust'].values
    throttle = motor_test_report['Throttle'].values
    current = motor_test_report['Current'].values
    # add 0 thrust and 0 throttle and current to the corresponding arrays
    thrust = np.append(0.0, thrust)
    throttle = np.append(0.0, throttle)
    current = np.append(0.0, current)
    print("At hover, throttle value is {:.2f} % and current consumption is {:.2f} A".format(np.poly1d(np.polyfit(thrust, throttle, degree))(thrust_value_at_hover), 4*np.poly1d(np.polyfit(thrust, current, degree))(thrust_value_at_hover)))


def plot_motor_test_report(motor_test_report, model):
    '''
    :param motor_test_report: dataframe of motor test report
    :param model: polynomial model of the motor test report
    :return: plot of the motor test report
    '''
    thrust = motor_test_report['Thrust'].values
    power = motor_test_report['Power'].values
    # add 0 thrust and 0 power to the corresponding arrays
    thrust_axis = np.linspace(0, thrust.max(), 100)
    # plot the real motor test report vs the modeled motor test report
    plt.plot(thrust, power, 'ro')
    plt.plot(thrust_axis, model(thrust_axis),'b')
    plt.title('Motor Test Report')
    plt.xlabel('Thrust [g]')
    plt.ylabel('Power [W]')
    plt.show()


def check_battery_cells(motor_test_report):
    '''
    :param motor_test_report: dataframe of motor test report
    :return: battery cells from the motor test report
    '''
    n_cells = int(np.floor(motor_test_report['Voltage'].str.replace(',', '.').astype(float).values/3.8).mean())
    print("Number of cells checked in motor test report: {}".format(n_cells))
    return n_cells


def max_hover_time(dry_mass, battery: Battery, motor_test_report, motor_mass):
    '''
    :param dry_mass: dry weight of the drone in grams (without motors, without battery).
    :param battery_cells: number of cells in the battery (6 for 6 cells - 6s battery, etc...)
    :param battery_capacity: milliampere-hours (mAh) of the battery
    :param battery_mass: mass of the battery in grams
    :param motor_test_report: dataframe of motor test report
    :param motor_mass: mass of a single motor in grams
    :return: maximum hover time in minutes
    '''
    total_mass = dry_mass + battery.mass + 4 * motor_mass
    thrust_needed_per_motor = total_mass / 4
    throttle_and_current_from_thrust(thrust_needed_per_motor, motor_test_report, degree=3)
    motor_model = model_motor_test_report(motor_test_report, degree=3) # Power(Thrust)
    motor_power = 4 * motor_model(thrust_needed_per_motor)
    battery_voltage = battery.cells * 3.7
    battery_power = battery_voltage * battery.capacity / 1000 # convert battery capacity to Ah -> W
    return 60 * battery_power / motor_power # in minutes


if __name__ == '__main__':
    path = r'C:\Users\omrijsharon\Downloads\2203_5_tmotors.csv'
    # path = r'C:\Users\omrijsharon\Downloads\1609397063572321.csv'
    motor_test_report = read_motor_test_report(path)
    motor_test_report_idx = 1
    dry_mass = 100
    battery = Battery(cells=6, capacity=3000, mass=304.2) #GEPRC VTC6 18650 6S1P 3000mAh
    battery_cells = check_battery_cells(motor_test_report[motor_test_report_idx]) # 6
    assert battery_cells == battery.cells, "Battery cells don't match"
    motor_mass = 19.7 #T-Motor 2203.5 1500KV
    hover_time = max_hover_time(dry_mass, battery, motor_test_report[1], motor_mass)
    print("Maximum hover time: {} minutes".format(hover_time))
    model = model_motor_test_report(motor_test_report[motor_test_report_idx], degree=4)
    plot_motor_test_report(motor_test_report[motor_test_report_idx], model)