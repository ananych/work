import numpy as np
from os import listdir
import os
from numpy import genfromtxt
import re

def read_csv(filename):
    """Objective: read_txt_files.
        @param filename: name of the input file.
        @return a numpy array contain all col of the file split by ','
    """
    my_data = genfromtxt(filename, delimiter=',')
    return my_data

def read_csv_with_time(filename):
    """Objective: read_txt_files.
        @param filename: name of the input file.
        @return a numpy array contain all col of the file split by ','
    """
    data = []
    start_time = []
    with open(filename, 'r', encoding='utf-8') as csv_data:
        time = csv_data.readline()
        if "time" in time:
            time = re.split(",", time)
            start_time.extend(list(map(float, time[1:])))
            #start_time[5] += start_time[6] / 1000000
            #start_time.pop()
        elif 'Start_Time' in time:
            time = time.split(':')[1]
            start_time.extend(list(map(float, time.split('-'))))        
        for time in csv_data.readlines():
            if ',' in time:
                time = time.split(',')[0] 
            if time == '':
                continue  
            data.append(float(time))

    return np.asarray(data), start_time

def get_file_list(path_data):
    """Objective: get file list in the path.
        @param path_data: the path of data.
        @return a list of file name
    """
    file_list = listdir(path_data)
    file_list_name = []
    for file in file_list:
        ext = os.path.splitext(file)[1]
        if ext == '.log' or ext == '.txt' or ext == '.csv':
            file_list_name.append(file)
    return file_list_name


def get_bitalino_bpm(filename):
    bpm = []
    start_time = []
    with open(filename, 'r', encoding='utf-8') as pressure_input:
        time = pressure_input.readline()
        if "time" in time:
            time = re.split(",|-", time)
            start_time.extend(list(map(float, time[1:])))
        else:
            log_data = time.split(',')
            bpm.append(int(log_data[0]))

        for line in pressure_input.readlines():
            log_data = line.split(',')
            bpm.append(int(log_data[0]))

    return np.array(bpm), start_time


def read_pressure_acc(path_data, filename):
    """Objective: get pressure data of the file.
        @param path_data: the path of data.
        @param filename: name of the input file.
        @param down_scale:down scale the sample rate of the file.
        @return a list of pressure
    """
    acc_x = []
    acc_y = []
    acc_z = []
    data = []
    start_time = []
    app_version = []
    with open(os.path.join(path_data, filename), 'r', encoding='utf-8') as pressure_input:
        #time = pressure_input.readline()
        for line in pressure_input.readlines():
            if "End Time" in line:
                break
            elif "State Change" in line:
                continue

            elif "Start Time" in line :
                time = re.split(":|/|-|,", line)
                start_time.extend(list(map(int, time[1:8])))
                start_time[5] += start_time[6] / 1000000
                start_time.pop()
            elif 'Start_Time' in line:
                time = line.split(':')[1]
                start_time.extend(list(map(int, time.split('-'))))  
            elif 'version' in line:
                app_version = line.split(':')[1]
            else:
                log_data = line.split(',')

                bcg = int(log_data[1].strip(' []'))

                data.append(bcg)
                acc_x.append(int(log_data[2].strip(' []')))
                acc_y.append(int(log_data[3].strip(' []')))
                acc_z.append(int(log_data[4].strip(' []')))


    return np.array(data), np.array(acc_x), np.array(acc_y), np.array(acc_z), start_time

def read_pressure_acc_result(path_data, filename, down_scale = 1):
    """Objective: get pressure data of the file.
        @param path_data: the path of data.
        @param filename: name of the input file.
        @param down_scale:down scale the sample rate of the file.
        @return a list of pressure
    """
    acc_x = []
    acc_y = []
    acc_z = []
    status = []
    conf_level = []
    HR = []
    data = []
    start_time = []
    timestamp = []
    app_version = []
    scale_count = 0
    with open(os.path.join(path_data, filename), 'r') as pressure_input:
        

        for line in pressure_input.readlines():
            if "End Time" in line:
                break
            elif "State Change" in line:
                continue

            elif "Start Time" in line :
                time = re.split(":|/|-|,", line)
                start_time.extend(list(map(int, time[1:8])))
                start_time[5] += start_time[6] / 1000000
                start_time.pop()
            elif 'Start_Time' in line:
                time = line.split(':')[1]
                start_time.extend(list(map(int, re.split(":|_|-|,", time))))  
            elif 'version' in line:
                app_version = line.split(':')[1]
                #line = line.split('.')
                #app_version = line
            else:
                if ',' in line:
                    log_data = line.split(',')
                else:
                    log_data = line.split('\t')
                # if len(log_data) > 5:
                bcg = int(log_data[1].strip(' []'))
                # if bcg >= pow(2,23):
                #     bcg = bcg % pow(2,23) 
                # else:
                #     bcg = bcg % pow(2,23) 
                #     bcg += pow(2,23) 
                scale_count += 1
                scale_count = scale_count % down_scale
                if scale_count == 0:                
                    data.append(bcg)
                    timestamp.append(int(log_data[0].strip(' []')))
                    acc_x.append(int(log_data[2].strip(' []')))
                    acc_y.append(int(log_data[3].strip(' []')))
                    acc_z.append(int(log_data[4].strip(' []')))
                    if len(log_data) >= 8:
                        status.append(int(log_data[7].strip(' []')))
                        HR.append(int(log_data[5].strip(' []')))

    return np.array(data), np.array(acc_x), np.array(acc_y), np.array(acc_z), np.array(status), np.array(HR),  np.array(timestamp), start_time


def read_time(path_data, filename):
    start_time = []
    with open(os.path.join(path_data, filename), 'r', encoding='utf-8') as pressure_input:
        time = pressure_input.readline()
        if "Start Time" in time:
            if ',' in time:
                time = time.split(',')[0]
            time = re.split(":|/|-", time)
            start_time.extend(list(map(int, time[1:])))
            start_time[5] += start_time[6] / 1000000
            start_time.pop()

    return start_time


def read_multi_sensor(path_data, filename):
    """Objective: get pressure data of the file.
        @param path_data: the path of data.
        @param filename: name of the input file.
        @param down_scale:down scale the sample rate of the file.
        @return a list of pressure
    """
    acc_data = []
    pressure_data = []
    start_time = []
    with open(os.path.join(path_data, filename), 'r', encoding='utf-8') as pressure_input:
        time = pressure_input.readline()
        if "Start Time" in time:
            time = re.split(":|/|-", time)
            start_time.extend(list(map(int, time[1:])))
            start_time[5] += start_time[6] / 1000000
            start_time.pop()
        else:
            pressure_input.seek(0, 0)

        for line in pressure_input.readlines():
            sensor = line.split(':')
            sensor_pressure_data = []
            sensor_acc_data = []

            for data in sensor:
                log_data = data.split(',')
                sensor_pressure_data.append(int(log_data[1]))
                sensor_acc_data.append(int(log_data[2]))
            pressure_data.append(sensor_pressure_data)
            acc_data.append(sensor_acc_data)

    return np.array(pressure_data).T, np.array(acc_data).T, start_time
