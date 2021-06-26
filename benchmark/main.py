import numpy as np
import random
import json
import diffprivlib

from pathlib import Path

ROOT_DIRECTORY = Path("/codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_GROUND_TRUTH = DATA_DIRECTORY / "ground_truth.csv"
DEFAULT_PARAMS = DATA_DIRECTORY / "parameters.json"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"

DEFAULT_PUBLIC = RUNTIME_DIRECTORY / 'public.csv'


def load_parameters(parameters_file):
    a ={}
    with parameters_file.open("r") as f:
        a = json.load(f)
    print(a["schema"]["shift"]["values"])
    return a


def load_ground_truth(ground_file):
    length = 0
    a = []
    with ground_file.open("r") as f:
        t = f.readline()
        for line in f:
            line = line.split(',')
            a.append(line)
            length = length + 1
    print(length)
    return a


# rows is raw data.

# load_ground_truth(rows_p, public_ground_truth_file)


def write_to_file(rows, file, epsilon):
    with file.open("a") as f:
        for row in rows:
            # check not to use the same taxi id twice
            a = (str(epsilon) + ',' + str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(row[3]) + ',' +
                 str(row[4]) + ',' + str(row[5]) + ',' +
                 str(row[8]) + ',' + str(row[9]) + ',' +
                 str(row[10]) + ',' + str(row[11]) + ',' + str(row[12]))
            f.write(a)
    f.close()




import statistics


def create_taxi_stats(rows, taxi_histogram):
    # sort list by taxi ID
    rows = np.array(rows)
    argsorted = np.argsort(rows[:, 0])
    rows = rows[argsorted, :]
    # 2 following lists for taxi feature summarization
    pickup_hist = []
    for i in range(0, 78):
        pickup_hist.append(0)
    averages_week = [0, 0, 0, 0, 0, 0, 0]
    i = -1
    j = 0
    total_taxis = 0
    for item in rows:
        i = i + 1

        # summarize information from taxi over all trips by that taxi
        if item[0] == rows[j][0]:
            if int(item[1]) == 0:
                day = 6
            else:
                day = int((int(item[1]) - 1) / 3)
            averages_week[int(day)] += 1

            pickup_hist[int(item[3])] += 1

            continue

        # taxi histogram keys. disjoint separators.
        average_temp = statistics.mean(averages_week)
        std_temp = statistics.stdev(averages_week)
        tot_trips = i - j
        cut_off1 = int(tot_trips / 2)
        cut_off2 = int(tot_trips / 3)
        cut_off3 = int(tot_trips / 4)
        combo1 = combo2 = combo3 = 0

        # key for average, std and pickup bin
        bin_avg = ''
        bin_std = ''
        bin_pickup = ''

        for q in taxi_histogram:
            k = int(q)
            if k < average_temp:
                continue
            bin_avg = q
            if k < 30:
                for w in taxi_histogram[q]:
                    l = int(w)
                    if l < std_temp:
                        continue
                    bin_std = w
                    break
            break
        if tot_trips > 35:
            for item2 in pickup_hist:
                if cut_off1 < item2:
                    combo1 += 1
                if cut_off2 < item2:
                    combo2 += 1
                if cut_off3 < item2:
                    combo3 += 1
            temp = 1000
            if combo1 == 1:
                temp += 100
            if combo2 >= 1:
                temp += 10
            if combo3 == 2:
                temp += 1

            bin_pickup = str(temp)
            bin_pickup = bin_pickup[1:]

        # add taxi ID to histogram
        taxi_id = rows[j][0]
        if int(bin_avg) < 30 and tot_trips > 35:
            taxi_histogram[bin_avg][bin_std][bin_pickup].append(taxi_id)
        elif int(bin_avg) == 30:
            taxi_histogram[bin_avg][bin_pickup].append(taxi_id)
        elif tot_trips < 35:
            taxi_histogram[bin_avg][bin_std].append(taxi_id)

        # clean up for next taxi

        j = i
        averages_week = [0, 0, 0, 0, 0, 0, 0]
        if int(item[1]) == 0:
            day = 6
        else:
            day = int((int(item[1]) - 1) / 3)
        averages_week[int(day)] += 1

        for k in range(0, 78):
            pickup_hist[k] = 0
        pickup_hist[int(item[3])] += 1


def sort_list(rows, column):
    rows = np.array(rows)
    argsorted = np.argsort(rows[:, column])
    return rows[argsorted, :]


def sample_from_public(taxi_histogram, public_taxi_hist, public_rows):
    public_rows = sort_list(public_rows, 0)
    pub_len = len(public_rows)
    synthetic = []

    print(len(np.unique(public_rows[:, 0])), len(public_rows))
    taxis = []
    drop_out_rate = 0.5
    random_taxi_id = 1000000
    for average_bin in taxi_histogram:
        if average_bin == '5':
            for std_bin in taxi_histogram[average_bin]:
                for item in range(0, len(taxi_histogram[average_bin][std_bin])):
                    # drop out if already chose this index
                    sample_index = random.randint(0, len(public_taxi_hist[average_bin][std_bin]) - 1)
                    sample_taxi_id = public_taxi_hist[average_bin][std_bin][sample_index]
                    while sample_taxi_id in taxis:
                        if random.random() < drop_out_rate:
                            break
                        sample_index = random.randint(0, len(public_taxi_hist[average_bin][std_bin]) - 1)
                        sample_taxi_id = public_taxi_hist[average_bin][std_bin][sample_index]

                    taxis.append(sample_taxi_id)
                    begin_index = np.searchsorted(public_rows[:, 0], sample_taxi_id)
                    i = begin_index
                    random_taxi_id += 1
                    while public_rows[begin_index][0] == public_rows[i][0]:
                        temp = public_rows[i].copy()
                        temp[0] = str(random_taxi_id)
                        synthetic.append(temp)
                        i += 1
                        if i >= pub_len:
                            break
        elif average_bin == '30':
            for pickup_bin in taxi_histogram[average_bin]:
                for item in range(0, len(taxi_histogram[average_bin][pickup_bin])):
                    sample_index = random.randint(0, len(public_taxi_hist[average_bin][pickup_bin]) - 1)
                    sample_taxi_id = public_taxi_hist[average_bin][pickup_bin][sample_index]
                    while sample_taxi_id in taxis:
                        if random.random() < drop_out_rate:
                            break
                        sample_index = random.randint(0, len(public_taxi_hist[average_bin][pickup_bin]) - 1)
                        sample_taxi_id = public_taxi_hist[average_bin][pickup_bin][sample_index]
                    taxis.append(sample_taxi_id)
                    begin_index = np.searchsorted(public_rows[:, 0], sample_taxi_id)
                    i = begin_index
                    random_taxi_id += 1

                    while public_rows[begin_index][0] == public_rows[i][0]:
                        temp = public_rows[i].copy()
                        temp[0] = str(random_taxi_id)
                        synthetic.append(temp)
                        i += 1
                        if i >= pub_len:
                            break
        else:
            for std_bin in taxi_histogram[average_bin]:
                for pickup_bin in taxi_histogram[average_bin][std_bin]:
                    for item in range(0, len(taxi_histogram[average_bin][std_bin][pickup_bin])):
                        sample_index = random.randint(0, len(public_taxi_hist[average_bin][std_bin][pickup_bin]) - 1)
                        sample_taxi_id = public_taxi_hist[average_bin][std_bin][pickup_bin][sample_index]
                        while sample_taxi_id in taxis:
                            if random.random() < drop_out_rate:
                                break
                            sample_index = random.randint(0,
                                                          len(public_taxi_hist[average_bin][std_bin][pickup_bin]) - 1)
                            sample_taxi_id = public_taxi_hist[average_bin][std_bin][pickup_bin][sample_index]
                        taxis.append(sample_taxi_id)
                        begin_index = np.searchsorted(public_rows[:, 0], sample_taxi_id)
                        i = begin_index
                        random_taxi_id += 1

                        while public_rows[begin_index][0] == public_rows[i][0]:
                            temp = public_rows[i].copy()
                            temp[0] = str(random_taxi_id)
                            synthetic.append(temp)
                            i += 1
                            if i >= pub_len:
                                break
    x = np.array(taxis)
    print(len(np.unique(x)), len(x))
    return synthetic


def apply_dp(taxi_histogram, myepsilon):
    mech1 = diffprivlib.mechanisms.GeometricTruncated().set_epsilon(.5).set_bounds(0,10000000).set_sensitivity(1)
    mech1.set_epsilon(myepsilon)

    for average_bin in taxi_histogram:
        if average_bin == '5':
            for std_bin in taxi_histogram[average_bin]:
                for i in range(0, len(taxi_histogram[average_bin][std_bin])):
                    temp = mech1.randomise(int(taxi_histogram[average_bin][std_bin][i]))
                    taxi_histogram[average_bin][std_bin][i] = temp
        elif average_bin == '30':
            for pickup_bin in taxi_histogram[average_bin]:
                for i in range(0, len(taxi_histogram[average_bin][pickup_bin])):
                    temp = mech1.randomise(int(taxi_histogram[average_bin][pickup_bin][i]))
                    taxi_histogram[average_bin][pickup_bin][i] = temp
        else:
            for std_bin in taxi_histogram[average_bin]:
                for pickup_bin in taxi_histogram[average_bin][std_bin]:
                    for i in range(0, len(taxi_histogram[average_bin][std_bin][pickup_bin])):
                        temp = mech1.randomise(int(taxi_histogram[average_bin][std_bin][pickup_bin][i]))
                        taxi_histogram[average_bin][std_bin][pickup_bin][i] = temp


if __name__ == '__main__':

    parameters_file2: Path = DEFAULT_PARAMS
    parms= {}
    parms = load_parameters(parameters_file2)
    myepsilon = 1.0
    mysensitivity = 200
    mytotalrecords = 20000000
    print('hi')
    rows_o = []
    ground_truth_file2: Path = DEFAULT_GROUND_TRUTH
    rows_o = load_ground_truth(ground_truth_file2)

    # for training
    import zipfile
    with zipfile.ZipFile('/codeexecution/submission/submission.zip', 'r') as zip_ref:
        zip_ref.extractall('/codeexecution/submission')
    rows_p = []
    public_ground_truth_file2: Path = DEFAULT_PUBLIC
    rows_p = load_ground_truth(public_ground_truth_file2)

    output_file: Path = DEFAULT_OUTPUT
    with output_file.open("w") as f:
        f.write(
            'epsilon,taxi_id,shift,company_id,pickup_community_area,dropoff_community_area,payment_type,fare,tips,'
            'trip_total,trip_seconds,trip_miles\n')

    # print('hi')
    for item in parms['runs']:
        myepsilon = (item['epsilon'])
        mysensitivity = (item['max_records_per_individual'])
        mytotalrecords = (item['max_records'])

        taxi_histogram_o = {'5': {'2': [], '200': []},
                            '11': {
                                '3': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []},
                                '6': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []},
                                '200': {'000': [], '001': [], '010': [], '011': [], '110': [],
                                        '111': []}},
                            '17': {
                                '6': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []},
                                '200': {'000': [], '001': [], '010': [], '011': [], '110': [],
                                        '111': []}},
                            '30': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []}}
        # public reference just a copy
        taxi_histogram_p = {'5': {'2': [], '200': []},
                            '11': {
                                '3': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []},
                                '6': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []},
                                '200': {'000': [], '001': [], '010': [], '011': [], '110': [],
                                        '111': []}},
                            '17': {
                                '6': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []},
                                '200': {'000': [], '001': [], '010': [], '011': [], '110': [],
                                        '111': []}},
                            '30': {'000': [], '001': [], '010': [], '011': [], '110': [], '111': []}}

        create_taxi_stats(rows_o, taxi_histogram_o)
        # print(taxi_histogram_o)
        apply_dp(taxi_histogram_o, myepsilon)

        create_taxi_stats(rows_p, taxi_histogram_p)

        synthetic_d = sample_from_public(taxi_histogram_o, taxi_histogram_p, rows_p)
        write_to_file(synthetic_d, output_file, myepsilon)
        # print(synthetic_d)
    #     temp = [1000001, 17, 50, 8, 8, 1, 6, 3, 10, 504, 0]
    #     synthetic_d = []
    #     synthetic_d.append(temp)
    #     synthetic_d.append(temp)
    #
    #     write_to_file(synthetic_d, output_file, myepsilon)
    #     write_to_file(synthetic_d, output_file, myepsilon * 10)
