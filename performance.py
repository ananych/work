# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:46:56 2019

@author: Blologue
"""

import numpy as np
from scipy.fftpack import fft


def remove_singular_value(signal_a, signal_b):
    """Objective: Remove the singular value -1 from signal_a and signal_b

    @param signal_a: Data a. Type: numpy array
    @param signal_b: Data b. Type: numpy array

    @return signal_a and signal_b after removal of -1
    """
    if signal_a.size > signal_b.size:
        signal_a = signal_a[:signal_b.size]
    elif signal_a.size < signal_b.size:
        signal_b = signal_b[:signal_a.size]
    keep_arr = np.logical_and(signal_a > 0, signal_b > 0)
    signal_c = signal_a[keep_arr]
    signal_d = signal_b[keep_arr]
    return signal_c, signal_d


def get_acc_nthu(test_signal, ground_truth):
    """Objective: Get the accurate rate by means of NTHU method
    Compute the average ratio of difference sample by sample between test_signal and ground_truth

    @param test_signal: Test data for heart beats. Type: numpy array
    @param ground_truth: Ground truth for heart beats. Type: numpy array

    @return The ratio as accuracy based on NTHU method
    """
    test, gt = remove_singular_value(test_signal, ground_truth)
    if test.size == 0:
        return 0
    error = abs(test - gt)
    acc_rate = np.mean(1 - error/gt)*100
    return round(acc_rate, 2)


def get_rms(signal_a, signal_b):
    """Objective: Get the statistic-related error of two data
    The error performance contains root mean square, mean absolute error, std of abs_error and abs_error itself

    @param signal_a: Data a. Type: numpy array
    @param signal_b: Data b. Type: numpy array

    @return The rms, mae, std, abs_error performance of two data
    """
    signal_c, signal_d = remove_singular_value(signal_a, signal_b)
    if signal_c.size == 0:
        return -1
    rms = round(np.sqrt(((signal_c - signal_d) ** 2).mean()), 2)
    abs_error = np.abs(signal_c - signal_d)
    mae = round(np.mean(abs_error), 2)
    std = round(np.std(abs_error), 2)
    return rms, mae, std, abs_error


def get_bpm_acc_rate(test_signal, ground_truth, bpm_error_tolerant):
    """Objective: Get the accurate rate by means of AP differences after removing singular value

    @param test_signal: Test data for heart beats. Type: numpy array
    @param ground_truth: Ground truth for heart beats. Type: numpy array
    @param bpm_error_tolerant: The error tolerant. Type: integer

    @return The APn as accuracy, n is error tolerant
    """
    test, gt = remove_singular_value(test_signal, ground_truth)
    if test.size == 0:
        return 0
    error = abs(test - gt)
    return round((np.sum(error <= bpm_error_tolerant) / len(test))*100, 2)


def get_acc(test_data, ground_truth, bpm_error_tolerant):
    '''Objective: Get the ratio of difference within bpm_error_tolerant
    Compute the ratio of bpm difference between test_data and ground_truth sample by sample within bpm_error_tolerant. The shorter is used to be total number of samples.

    @param test_data: Test data for heart beats. Type: numpy array
    @param ground_truth: Ground truth for heart beats. Type: numpy array
    @param bpm_error_tolerant: Error tolerant for bpm. Type: integer

    @return The ratio as accuracy
    '''
    if len(ground_truth) > len(test_data):
        ground_truth = ground_truth[:len(test_data)]
    else:
        test_data = test_data[:len(ground_truth)]

    if len(test_data) == 0:
        return -1
    diff = np.abs(test_data - ground_truth)
    if len(test_data) == 0:
        acc = round(np.sum(diff <= bpm_error_tolerant) / 1, 4) * 100
    else:
        acc = round(np.sum(diff <= bpm_error_tolerant) / len(test_data), 4) * 100

    return acc


def get_acc_HRV_scale(test_data, ground_truth, bpm_error_tolerant):
    '''Objective: Get the ratio of difference within bpm_error_tolerant
    Compute the ratio of bpm difference between test_data and ground_truth sample by sample within bpm_error_tolerant. The shorter is used to be total number of samples.

    @param test_data: Test data for heart beats. Type: numpy array
    @param ground_truth: Ground truth for heart beats. Type: numpy array
    @param bpm_error_tolerant: Error tolerant for bpm. Type: integer

    @return The ratio as accuracy
    '''
    if len(ground_truth) > len(test_data):
        ground_truth = ground_truth[:len(test_data)]
    else:
        test_data = test_data[:len(ground_truth)]

    if len(test_data) == 0:
        return -1
    start = 0
    window = 60 * 5
    abs_error = []
    while (start + window) < len(test_data):
        temp = test_data[start : start + window]
        temp2 = ground_truth[start : start + window]
        abs_error.append(np.abs(np.median(temp) - np.median(temp2)))
        start += 60
    abs_error = np.array(abs_error)
    if len(abs_error) == 0:
        return -1 , []
        
    if len(test_data) == 0:
        acc = round(np.sum(abs_error <= bpm_error_tolerant) / 1, 4) * 100
    else:
        acc = round(np.sum(abs_error <= bpm_error_tolerant) / len(abs_error), 4) * 100

    return acc, abs_error

def performance(test_data, ground_truth, bpm_error_tolerant=5, HRV_scale=0):
    """Objective: Evaluate the performance of test_data based on ground_truth
    Call some accuracy API to get performance.

    @param test_data: Test data for heart beats. Type: numpy array
    @param ground_truth: Ground truth for heart beats. Type: numpy array
    @param bpm_error_tolerant: Error tolerant for bpm, default is 5. Type: integer

    @return The AP rate, rms, mae, std and abs_error as accuracy
    """   
    rms_value = get_rms(test_data, ground_truth)
    if rms_value == -1:
        return -1, -1, -1, -1, -1
    else:
        rms, mae, std, abs_error = rms_value
    if HRV_scale == 0:
        acc_rate = get_acc(test_data, ground_truth, bpm_error_tolerant)
    else:
        acc_rate, abs_error = get_acc_HRV_scale(test_data, ground_truth, bpm_error_tolerant)

    return acc_rate, rms, mae, std, abs_error


def detectable_rate(bpm):
    '''Compute the ratio of detectable samples to all samples
    The detectable sample mean that the sample's value is not -1.

    @param bpm: Test data for heart beats. Type: numpy array

    @return The ratio of detectable samples
    '''
    counter = np.sum(bpm == -1)
    detectable = round((1 - counter/len(bpm))*100, 2)

    return detectable


# Currently, this API is not used
def beats_to_sec(bpm, bpm_pos=[]):
    """Objective: change bpm x axis from beats to sec
        @param bpm_pos: position of beats
        @param bpm: heart beats per min
        @return bpm : a numpy array of bpm every sec
    """
    if len(bpm_pos) > 0:
        bpm_time = (bpm_pos - bpm_pos[0]) / 60
        bpm_time = bpm_time[1:]
    else:
        bpm_time = 60 / bpm
        for i in range(1,len(bpm_time)):
            bpm_time[i] += bpm_time[i-1]

    total_time = int(bpm_time[-1])

    return np.interp(np.arange(1,total_time + 1), bpm_time, bpm)

# Currently, this API is not used
def get_optimal_acc_rate(test_signal, ground_truth, bpm_error_tolerant=5, max_shift=10):
    """Objective: Get the optimal outcome
        @param test_signal: a numpy array of bpm test data
        @param ground_truth: a numpy array of bpm ground truth
        @param bpm_error_tolerant: bpm error tolerant range, default is 5
        @param max_shift: shift range to align data, default is 10
        @return accurate rate and data after time align
    """
    exchange = False
    if test_signal.size > ground_truth.size:
        temp = ground_truth
        ground_truth = test_signal
        test_signal = temp
        exchange = True
    optimal_shift = 0
    max_acc_rate = get_bpm_acc_rate(test_signal, ground_truth[:test_signal.size], bpm_error_tolerant)
    for i in range(1, max_shift + 1):
        #left shift
        acc_rate = get_bpm_acc_rate(test_signal[i:], ground_truth[:test_signal.size - i], bpm_error_tolerant)
        if acc_rate > max_acc_rate:
            max_acc_rate = acc_rate
            optimal_shift = -i
        #right shift
        if test_signal.size + i <= ground_truth.size:
            acc_rate = get_bpm_acc_rate(test_signal, ground_truth[i:test_signal.size + i], bpm_error_tolerant)
            if acc_rate > max_acc_rate:
                max_acc_rate = acc_rate
                optimal_shift = i
        else:
            acc_rate = get_bpm_acc_rate(test_signal[:ground_truth.size - i], ground_truth[i:], bpm_error_tolerant)
            if acc_rate > max_acc_rate:
                max_acc_rate = acc_rate
                optimal_shift = i

    if optimal_shift > 0:
        optimal_ground_truth = ground_truth[optimal_shift: test_signal.size+optimal_shift]
        if exchange:
            return -optimal_shift, optimal_ground_truth, test_signal[:optimal_ground_truth.size]
        else:
            return optimal_shift, test_signal[:optimal_ground_truth.size], optimal_ground_truth
    else:
        optimal_test_signal = test_signal[-optimal_shift: ground_truth.size - optimal_shift]
        if exchange:
            return -optimal_shift, ground_truth[:optimal_test_signal.size], optimal_test_signal
        else:
            return optimal_shift, optimal_test_signal, ground_truth[:optimal_test_signal.size]

def align_results(time_sum, bpm, ground_truth_bpm, state, confidence_level, error_tolerant):
    time_shift_sum = int(round(time_sum)) + 3
    if time_sum == 0:
        shift, test_data, ground_truth = get_optimal_acc_rate(bpm, ground_truth_bpm, error_tolerant)
        if shift > 0:
            state = state[:len(test_data)]
            confidence_level = confidence_level[:len(test_data)]
        else:
            state = state[-shift:len(test_data) - shift]
            confidence_level = confidence_level[-shift:len(test_data) - shift]
    else:
        if time_shift_sum > 0:
            ground_truth = ground_truth_bpm[time_shift_sum:len(bpm) + time_shift_sum]
            test_data = bpm[:len(ground_truth)]
            state = state[:len(test_data)]
            confidence_level = confidence_level[:len(test_data)]
        else:
            test_data = bpm[-time_shift_sum:len(ground_truth_bpm) - time_shift_sum]
            state = state[-time_shift_sum:len(test_data) - time_shift_sum]
            ground_truth = ground_truth_bpm[:len(test_data)]
            confidence_level = confidence_level[-time_shift_sum:len(test_data) - time_shift_sum]

    state = state[test_data != -1]
    ground_truth = ground_truth[test_data != -1]
    confidence_level = confidence_level[test_data != -1]
    test_data = test_data[test_data != -1]

    return test_data, ground_truth, state, confidence_level

# Currently, this API is not used
def get_coverage(test_data, ground_truth):
    '''Objective: Get the coverage of raw data w.r.t. ground truth
    Compute the ratio of the number of samples in test_data compare to ground_truth

    @param test_data: Test data for heart beats. Type: numpy array
    @param ground_truth: Ground truth for heart beats. Type: numpy array

    @return The coverage as accuracy
    '''
    return (1-(abs(len(test_data)-len(ground_truth))/len(ground_truth)))*100


# Currently, this API is not used
def perf_without_gt(bpm_without_post, bpm):
    """Objective: Evaluate the performance without ground truth
    The bpm with postprocessing (get median between 5 adjacent samples) is used as ground truth. The first performance metric is AP10 by default. The second performance metric is the ratio of adjacent difference larger than 10 by default compare to the previous sample.

    @param bpm_without_post: bpm without post process. Type: numpy array
    @param bpm: bpm with post process. Type: numpy array

    @return The AP10 and the ratio of adjacent difference larger than 10 by default as accuracy
    """
    bpm_smooth = np.copy(bpm_without_post)
    for i in range(2, len(bpm_without_post) - 3):
        bpm_smooth[i] = np.median(bpm_without_post[i - 2:i + 3])
    acc = get_bpm_acc_rate(bpm_without_post,bpm_smooth,10)
    error = 0
    for i in range(1, bpm.size):
        if bpm[i] - bpm[i-1] > 10:
            error += 1

    error = round(error / (bpm.size - 1) * 100, 2)

    return acc, error


def main():
    # As test data for heart beats
    a = np.array([61, 65, 63, 70, 83, -1, 85, 76, 71, 72])
    # As ground truth for heart beats
    b = np.array([69, 71, 62, 71, 82, 90, -1, 75, 72, 67])

    # Test API remove_singular_value
    test_a, test_b = remove_singular_value(a, b)
    correct_a = np.array([61, 65, 63, 70, 83, 76, 71, 72])
    correct_b = np.array([69, 71, 62, 71, 82, 75, 72, 67])
    assert np.array_equal(test_a, correct_a) and np.array_equal(test_b, correct_b), 'API remove_singular_value error!'

    # Test API get_acc_nthu
    test_accuracy = get_acc_nthu(a, b)
    assert round(test_accuracy, 2) == 95.69, 'API get_acc_nthu error!'

    # Test API get_rms
    test_rms, test_mae, test_std, _ = get_rms(a, b)
    assert test_rms == 4.03 and test_mae == 3.0 and test_std == 2.69, 'API get_rms error!'

    # Test API get_bpm_acc_rate
    test_AP10 = get_bpm_acc_rate(a, b, 10)
    test_AP5 = get_bpm_acc_rate(a, b, 5)
    assert test_AP10 == 100.00 and test_AP5 == 75.00, 'API get_bpm_acc_rate error!'

    # Test API get_acc
    test_AP10 = get_acc(a, b, 10)
    test_AP5 = get_acc(a, b, 5)
    assert test_AP10 == 80.00 and test_AP5 == 60.00, 'API get_acc error!'

    # Test API performance
    test_AP10, test_rms, test_mae, test_std, _ = performance(a, b, 10)
    assert test_AP10 == 80.00 and test_rms == 4.03 and test_mae == 3.0 and test_std == 2.69, 'API performance error!'

    # Test API detectable_rate
    test_a = detectable_rate(a)
    test_b = detectable_rate(b)
    assert test_a == 90.00 and test_b == 90.00, 'API detectable_rate error!'


if __name__ == '__main__':
    main()