import ast
import warnings
import time # timer
import psutil # cpu/mem/IO usage
import heapq # 1D tree
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import editdistance
from geopy.distance import great_circle
from sklearn.neighbors import BallTree  # KD-tree
from shapely.geometry import LineString, Point, box
from shapely import wkt
from paretoset import paretoset # skyline query
from rtree import index  # rtree

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Useful functions
def great_circle_dist_line(line: LineString) -> int:
    '''
    Helper function that calculates the distance of the linestring in great circle distance.
    '''
    total_distance = 0
    coords = list(line.coords)
    for index, coord in enumerate(coords):
        if index + 1 < len(coords):
            next_coord = coords[index+1]
            total_distance += great_circle(coord, next_coord).km

    return total_distance

def visualise(num=3):
    '''
    A helper function that plots all of the taxi's trip.
    '''
    df = pd.read_csv(
        f'Databases/Jul-0{num}.csv').iloc[:, 1:].dropna(subset=['linestring']).reset_index()
    df['geom'] = df.linestring.apply(wkt.loads)  # convert string to linestring
    # create geo dataframe
    gdf = gpd.GeoDataFrame(df['index'], geometry=df['geom'])

    gdf.plot()
    plt.show()

def compare_lists(actual: list, predicted: list):
    '''
    Calculates the precision, recall and fmeasure given two lists.
    '''
    true_positives = len(set(actual) & set(predicted))
    false_negatives = len(set(actual) - set(predicted))
    false_positives = len(set(predicted) - set(actual))
    
    if true_positives + false_positives != 0:
        precision = true_positives / float(true_positives + false_positives)
    else:
        precision = 0
    
    if true_positives + false_negatives != 0:
        recall = true_positives / float(true_positives + false_negatives)
    else:
        recall = 0
        
    if precision + recall != 0:
        fmeasure = (2 * precision * recall)/float(precision + recall)
    else:
        fmeasure = 0

    # print('Precision:', precision)
    # print('Recall:', recall)
    # print('f-measure:', fmeasure)

    return precision, recall, fmeasure

def compare_dicts(actual: dict, predicted: dict):
    '''
    Calculates the precision, recall and fmeasure given two dictionaries.
    '''
    true_positives = 0
    possible_true_positives = set(actual.keys()) & set(predicted.keys())
    false_negatives = len(set(actual) - set(predicted))
    false_positives = len(set(predicted) - set(actual))

    # check if values are equal
    for key in possible_true_positives:
        if sorted(actual[key]) == sorted(predicted[key]):
            true_positives +=1
        else:
            false_positives += 1

    precision = true_positives / float(true_positives + false_positives)
    recall = true_positives / float(true_positives + false_negatives)
    fmeasure = (2 * precision * recall) / float(precision + recall)

    # print('Precision:', precision)
    # print('Recall:', recall)
    # print('f-measure:', fmeasure)

    return precision, recall, fmeasure


# Query 1
bboxes = [
    (-8.59, 41.15, -8.57, 41.17),
    (-8.45, 41.15, -7.7, 41.5),
    (-8.6, 41.2, -7.9, 41.7),
    (-8.8, 41.2, -8, 42),
    (-8.6, 41.2, -8, 41.5),
]

true_values1 = [1372660161620000403,
                1372666820620000446,
                1372669161620000195,
                1372672070620000017,
                1372673370620000324,
                1372675292620000131,
                1372679366620000403,
                1372677491620000620,
                1372683647620000311,
                1372697835620000041,
                1372710883620000545,
                1372721836620000006]

true_values2 = [1372764603620000410]

true_values3 = [1372779612620000129]

true_values4 = [1372873397620000596,
                1372875022620000596,
                1372880481620000080,
                1372885275620000901,
                1372912453620000160,
                1372908827620000596,
                1372909210620000434,
                1372922111620000562,
                1372926281620000562,
                1372872446620000192]

true_values5 = [1372964669620000010,
                1372996755620000197,
                1373023121620000458]

true_values = [true_values1, true_values2, true_values3, true_values4, true_values5]

def q1a():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()

    precision_list = []
    recall_list = []
    fmeasure_list = []
    gdf_list = []
    
    for i, bbox in enumerate(bboxes):
        contains = []
        df = pd.read_csv(
            f'Databases/Jul-0{i+1}.csv').iloc[:, 1:].dropna(subset=['linestring']).reset_index()
        # convert string to linestring
        df['geom'] = df.linestring.apply(wkt.loads)
        # create geo dataframe
        gdf = gpd.GeoDataFrame(df['index'], geometry=df['geom'])

        # rtree
        trips = gdf.sindex.query(box(bbox[0],
                                     bbox[1],
                                     bbox[2],
                                     bbox[3]),
                                 predicate='contains')

        # print(f'----------July-{i+1}----------')
        
        geom_list = []
        for trip in trips:
            contains.append(df.at[trip, 'trip_id'])
            geom_list.append(df.at[trip, 'geom'])
        
        gdf_list.append(gpd.GeoDataFrame(geometry=geom_list))

        precision, recall, fmeasure = compare_lists(true_values[i], contains)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)

    # Load the GeoDataFrames
    gdf1 = gdf_list[0]
    gdf2 = gdf_list[1]
    gdf3 = gdf_list[2]
    gdf4 = gdf_list[3]
    gdf5 = gdf_list[4]

    # Create a figure with 5 subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    # Plot each GeoDataFrame on a separate subplot
    gdf1.plot(ax=axs[0])
    gdf2.plot(ax=axs[1])
    gdf3.plot(ax=axs[2])
    gdf4.plot(ax=axs[3])
    gdf5.plot(ax=axs[4])

    # Set the title for each subplot
    axs[0].set_title("July-1")
    axs[1].set_title("July-2")
    axs[2].set_title("July-3")
    axs[3].set_title("July-4")
    axs[4].set_title("July-5")
    # _, ax = plt.subplots()
    # ax.set_xticklabels(['Precision', 'Recall', 'F-measure'])
    # ax.set_ylim([-0.1, 1.1])
    
    # data = [precision_list, recall_list, fmeasure_list]
    # ax.boxplot(data)
    # plt.show()
    
    # benchmark
    end = time.time()
    print('R-tree stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

def q1b():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []
    for i, bbox in enumerate(bboxes):
        contains = []
        df = pd.read_csv(
            f'Databases/Jul-0{i+1}.csv').iloc[:, 1:].dropna(subset=['linestring']).reset_index()
        # convert polyline string to list
        df['polyline'] = df.polyline.apply(ast.literal_eval)

        # perform linear search
        # print(f'----------{i+1}----------')
        for row, polyline in enumerate(df['polyline']):
            x = [lst[0] for lst in polyline]
            y = [lst[1] for lst in polyline]
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            max_y = max(y)

            if min_x >= bbox[0] and max_x <= bbox[2] \
                    and min_y >= bbox[1] and max_y <= bbox[3]:
                contains.append(df.at[row, 'trip_id'])
        
        precision, recall, fmeasure = compare_lists(true_values[i], contains)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)

    # _, ax = plt.subplots()
    # ax.set_xticklabels(['Precision', 'Recall', 'F-measure'])
    # ax.set_ylim([-0.1, 1.1])
    
    # data = [precision_list, recall_list, fmeasure_list]
    # ax.boxplot(data)
    end = time.time()
    print('\nLinear search stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))
    # plt.show()


# Query 2 - sim trajectories
true_values1 = [1372688474620000049,
                1372676788620000245,
                1372650711620000403,
                1372691971620000239,
                1372683466620000189]

true_values2 = [1372724111620000112,
                1372688474620000049,
                1372691971620000239,
                1372765248620000196,
                1372696625620000624]

true_values3 = [1372794890620000030,
                1372840713620000257,
                1372850227620000697,
                1372782866620000570,
                1372802139620000496]

true_values4 = [1372945414620000349,
                1372932133620000451,
                1372909163620000463,
                1372894195620000671,
                1372945995620000095]

true_values5 = [1372949023620000011,
                1372949181620000148,
                1372949567620000086,
                1372956462620000053,
                1372992883620000612]

true_values = [true_values1, true_values2, true_values3, true_values4, true_values5]

strings = ['0.16,52|0.5,35|0.15,-18|0.3,-16',
           '0.9,50|0.4,-50|0.5,-12|0.05,10',
           '0.04,50|0.1,30|0.3,-30|0.2,52|0.5,-12|0.1,10|0.6,10',
           '0.43,32|0.52,90|-0.2,-35',
           '0.1,-10|0.4,32|0.2,-102|0.12,13|0.43,10']

def to_linestring(string, start=(0,0)):
    '''
    helper function
    '''
    import great_circle_calculator.great_circle_calculator as gcc
    line_list = []
    line_list.append(start)
    seq = string.split('|')
    
    x, y = start[0], start[1]
    
    for coord in seq:
        bearing, distance = coord.split(',')
        # x, y = calculate_destination_point(y, x, float(bearing), float(distance))
        x, y = gcc.point_given_start_and_bearing(start, float(bearing), float(distance))
        
        line_list.append((x, y))
    
    return LineString(line_list)
    
def q2a():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []
    gdf_list = []

    for i in range(5):
        # print(f'----------{i+1}----------')
        df = pd.read_csv(f'Sim_trajectories/sim{i+1}.csv')
        trajectory = strings[i]
        result = []
        line_list = [] # graphing purposes

        # if edit distance is greater than the lowest edit distance, skip the row
        lowest_distance = 10000

        # linear search
        for col1, str1 in enumerate(df['to_string']):
            if not isinstance(str1, float):  # check if not empty string
                length_diff = abs(len(str1) - len(trajectory))
                # dont check if diff between 2 strings is less than lowest distance
                if (length_diff > lowest_distance) or len(str1) > 255:
                    pass
                else:
                    edit_dist = editdistance.eval(str1, trajectory)
                    if edit_dist <= lowest_distance:  # if new lower distance have been found
                        # result.append((edit_dist, col1))
                        result.append(df.at[col1, 'trip_id'])
                        lowest_distance = edit_dist
                        line_list.append(df.at[col1, 'linestring'])

        similar_line = wkt.loads(line_list[-1])
        actual_line = to_linestring(trajectory, similar_line.coords[0])

        gdf_list.append(gpd.GeoDataFrame(geometry=[similar_line, actual_line]))

        # bench mark
        precision, recall, fmeasure = compare_lists(true_values[i], result[-5:])
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)
        
    end = time.time()
    print('\nLinear search stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

    # _, ax = plt.subplots()
    # ax.set_xticklabels(['Precision', 'Recall', 'F-measure'])
    # ax.set_ylim([-0.1, 1.1])
        
    # data = [precision_list, recall_list, fmeasure_list]
    # ax.boxplot(data)
    
    # Load the GeoDataFrames
    # gdf1 = gdf_list[0]
    # gdf2 = gdf_list[1]
    # gdf3 = gdf_list[2]
    # gdf4 = gdf_list[3]
    # gdf5 = gdf_list[4]

    # Create a figure with 5 subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    # Plot each GeoDataFrame on a separate subplot
    # gdf1.plot(ax=axs[0], color=['blue', 'red'])
    # gdf2.plot(ax=axs[1], color=['blue', 'red'])
    # gdf3.plot(ax=axs[2], color=['blue', 'red'])
    # gdf4.plot(ax=axs[3], color=['blue', 'red'])
    # gdf5.plot(ax=axs[4], color=['blue', 'red'])

    # Set the title for each subplot
    # axs[0].set_title("July-1")
    # axs[1].set_title("July-2")
    # axs[2].set_title("July-3")
    # axs[3].set_title("July-4")
    # axs[4].set_title("July-5")

    # plt.show()

def q2b():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []
    for i in range(5):
        # print(f'----------{i+1}----------')
        df = pd.read_csv(f'Sim_trajectories/sim{i+1}.csv')
        trajectory = strings[i]
        hq = []
        
        for row, line in enumerate(df['to_string']):
            if isinstance(line, str):
                if len(line) < 255:
                    edit_distance = editdistance.eval(line, trajectory)
                    heapq.heappush(hq, (edit_distance, df.at[row, 'trip_id']))
        
        result = []
        for _ in range(5):
            result.append(heapq.heappop(hq)[1])
        compare_lists(true_values[i], result)
        precision, recall, fmeasure = compare_lists(true_values[i], result)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)

    end = time.time()
    print('\nKD-Tree stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))


    # _, ax = plt.subplots()
    # ax.set_xticklabels(['Precision', 'Recall', 'F-measure'])
    # ax.set_ylim([-0.1, 1.1])
        
    # data = [precision_list, recall_list, fmeasure_list]
    # ax.boxplot(data)

    # plt.show()


# Query 3 - knn
points = [((-8.613, 41.145), 4),
          ((-8.2, 41.3), 3),
          ((-8.1, 41.5), 2),
          ((-8.4, 41.7), 3),
          ((-8.3, 41.3), 3)]

true_values1 = [1372636951620000320,
                1372690799620000516,
                1372693194620000269,
                1372710782620000473]

true_values2 = [1372758899620000178,
                1372761844620000351,
                1372769858620000131]

true_values3 = [1372809455620000616,
                1372781638620000250]

true_values4 = [1372934636620000178,
                1372926281620000562,
                1372922111620000562]

true_values5 = [1373002337620000338,
                1373016132620000159,
                1372970338620000245]

true_values = [true_values1, true_values2,
               true_values3, true_values4, true_values5]

def q3a():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()

    precision_list = []
    recall_list = []
    fmeasure_list = []
    
    for i, p in enumerate(points):
        df = pd.read_csv(
            f'Databases/Jul-0{i+1}.csv').dropna(subset=['linestring']).iloc[:, 1:]
        df = df.reset_index(drop=True).reset_index()
        # convert string to linestring
        df['geom'] = df.linestring.apply(wkt.loads)
    
        gdf = gpd.GeoDataFrame(df, geometry=df['geom'])  # create geo dataframe
        
        # perform knn
        point = Point(p[0])
        result = []
        # print(f'----------{i+1}----------')
        for _ in range(p[1]):
            index, _ = gdf.sindex.nearest(point, return_distance=True)  # create rtree
            trip_id = gdf.at[index[1][0], 'trip_id']
            result.append(trip_id)  # add shortest trip to the list

            # reformat dataframe
            gdf = gdf.drop([index[1][0]])  # remove the shortest distance
            gdf = gdf.iloc[:, 1:]
            gdf = gdf.reset_index(drop=True)  # reset the index
            gdf = gdf.reset_index()

        precision, recall, fmeasure = compare_lists(true_values[i], result)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)

    end = time.time()
    print('\nR-tree stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

    plt.show()

def q3b():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []

    for i, p in enumerate(points):
        df = pd.read_csv(
            f'Databases/Jul-0{i+1}.csv').dropna(subset=['linestring']).iloc[:, 1:]
        df = df.reset_index(drop=True).reset_index()
        df['polyline'] = df.polyline.apply(ast.literal_eval)

        fixed_point = Point(p[0])
        distance_list = []

        # linear search
        # print(f'----------{i+1}----------')
        for row, polyline in enumerate(df['polyline']):
            temp_dist = []
            for coord in polyline:
                point1 = Point(coord)
                temp_dist.append(great_circle(fixed_point.coords, point1.coords).m)
                # temp_dist.append(point1.distance(fixed_point))
                
            heapq.heappush(distance_list, (min(temp_dist), df.at[row, 'trip_id']))

        result = []

        for _ in range(p[1]):
            dist, trip_id = heapq.heappop(distance_list)
            # print(dist)
            result.append(trip_id)
        
        precision, recall, fmeasure = compare_lists(true_values[i], result)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)
        
    end = time.time()
    print('\nLinear search stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

    # _, ax = plt.subplots()
    # ax.set_xticklabels(['Precision', 'Recall', 'F-measure'])
    # ax.set_ylim([-0.1, 1.1])
        
    # data = [precision_list, recall_list, fmeasure_list]
    # ax.boxplot(data)

    # plt.show()


# Query 4 - skyline
true_values1 = [(0.0009893219246441792,30),
                (0.0009893364929513084,15)]

true_values2 = [(0.0009893364929513084,15)]

true_values3 = [(0.000989541065821645,15)]

true_values4 = [(0.002001511507345703,30),
                (0.0022176518739589163,15)]

true_values5 = [(0.0009893430756530304,30),
                (0.0009894170750687741,15)]

true_values = [true_values1, true_values2,
               true_values3, true_values4, true_values5]

def q4a():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []
    
    fig, ax = plt.subplots()

    for i in range(5):
        # print(f'---------{i+1}---------')
        df = pd.read_csv(f'Databases/Jul-0{i+1}.csv').iloc[:, 1:]

        # convert polyline string to list
        df['polyline'] = df.polyline.apply(ast.literal_eval)
        new_df = pd.DataFrame()
        new_df['trip_id'] = 0
        new_df['travel'] = 0
        new_df['time'] = 0
        new_df['geom'] = Point(0, 0)

        # drop unfinished trips
        non_travel = []
        for col, line in enumerate(df['polyline']):
            if len(line) <= 1:  # find unfinished travel
                non_travel.append(col)

            elif LineString(line).length == 0:  # remove when travelled for 0 distance
                non_travel.append(col)

            else:
                new_df.at[col, 'trip_id'] = df.at[col, 'trip_id']
                new_df.at[col, 'travel'] = great_circle_dist_line(LineString(line))
                new_df.at[col, 'time'] = (len(line) - 1) * 15
                new_df.at[col, 'geom'] = Point((len(line) - 1) * 15, great_circle_dist_line(LineString(line)))
                
        gpd.GeoDataFrame(geometry=new_df['geom']).plot(ax=ax, color='blue') #scatterplot
        new_df = new_df.reset_index().iloc[:, 1:-1]
        
        # perform NN using rtree to find skyline points
        skyline_queries = []
        nearest = (0, 0)
        while not new_df.empty:
            # construct rtree
            rtree = index.Index()
            for col, travel_time in enumerate(new_df.values):
                rtree.insert(col, travel_time[1:])

            nearest_index_list = list(rtree.nearest((nearest[0], nearest[1])))
            nearest_coords = new_df.iloc[nearest_index_list[0], 1:]

            x = nearest_coords[0]
            y = nearest_coords[1]
            nearest = (x, y)

            skyline_queries.append(nearest)
            dominated = list(rtree.intersection((x, y, np.inf, np.inf)))
            dominated += nearest_index_list

            new_df = new_df.drop(dominated).reset_index(drop=True)

        # print(skyline_queries)
        lst = []
        for coord in skyline_queries:
            lst.append(Point(coord[1], coord[0]))
        gpd.GeoDataFrame(geometry=lst).plot(ax=ax, color='red')
        
        precision, recall, fmeasure = compare_lists(true_values[i], skyline_queries)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)
        break
        
    end = time.time()
    print('R-tree stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

    # _, ax = plt.subplots()
    # ax.set_xticklabels(['Precision', 'Recall', 'F-measure'])
    # ax.set_ylim([-0.1, 1.1])
        
    # data = [precision_list, recall_list, fmeasure_list]
    # ax.boxplot(data)    

    plt.xlim(0, 14000)
    plt.legend(['trips', 'skyline point'])
    plt.xlabel('Time')
    plt.ylabel('Distance travelled')
    plt.show()

def q4b():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []

    for i in range(5):
        # print(f'---------{i+1}---------')
        df = pd.read_csv(f'Databases/Jul-0{i+1}.csv').iloc[:, 1:]
        # convert polyline string to list
        df['polyline'] = df.polyline.apply(ast.literal_eval)
        new_df = pd.DataFrame()
        new_df['trip_id'] = 0
        new_df['travel'] = 0
        new_df['time'] = 0

        # drop unfinished trips
        non_travel = []
        for col, line in enumerate(df['polyline']):
            if len(line) <= 1:  # find unfinished travel
                non_travel.append(col)

            elif LineString(line).length == 0:  # remove when travelled for 0 distance
                non_travel.append(col)

            else:
                new_df.at[col, 'trip_id'] = df.at[col, 'trip_id']
                new_df.at[col, 'travel'] = LineString(line).length
                new_df.at[col, 'time'] = (len(line) - 1) * 15
        new_df = new_df.reset_index().iloc[:, 2:]

        # perform skyline query
        mask = paretoset(new_df, sense=["min", "min"])
        skyline_queries = []

        for row, bol in enumerate(mask):
            if bol:
                skyline_queries.append(
                    (new_df.at[row, 'travel'], new_df.at[row, 'time']))

        # compare_lists(true_values[i], skyline_queries)
        precision, recall, fmeasure = compare_lists(true_values[i], skyline_queries)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)
        
    end = time.time()
    print('\nPareto stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

    # _, ax = plt.subplots()
    # ax.set_xticklabels(['Precision', 'Recall', 'F-measure'])
    # ax.set_ylim([-0.1, 1.1])
        
    # data = [precision_list, recall_list, fmeasure_list]
    # ax.boxplot(data)

    # plt.show()


# Query 5 - closest starting points
def q5a():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []
    
    for i in range(5):
        # print(f'---------{i+1}---------')
        df = pd.read_csv(f'Databases/Jul-0{i+1}.csv').iloc[:, 1:]
        df['geom'] = df.polyline.apply(ast.literal_eval)

        new_df = pd.DataFrame(columns=['x', 'y'])
        x_list = []
        y_list = []

        for line in df['geom']:
            if line != []:
                x_list.append(line[0][0])
                y_list.append(line[0][1])
            else:
                x_list.append(0)
                y_list.append(0)

        new_df['x'] = x_list
        new_df['y'] = y_list

        # search with KD-tree
        result = {}
        visited = set()
        kd_tree = BallTree(new_df, metric='haversine')

        for key_index, row in new_df.iterrows():
            x = row['x']
            y = row['y']
            if x == 0 and y == 0:
                pass
            else:
                ind = (kd_tree.query_radius([[x,y]], r=0.0000001))[0].tolist()
                ind.remove(key_index)
                
                # add to results if there are near neighbours
                if ind != [] and key_index not in visited:
                    result.update({key_index: ind})
                    visited.update(ind)

        df_true_values = pd.read_csv(f'Start_points/true_values{i+1}.csv')
        true_values = df_true_values.groupby('point_a_id')['point_b_id'].apply(list).to_dict()
        
        precision, recall, fmeasure = compare_lists(true_values, result)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)

    end = time.time()
    print('KD-tree stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

def q5b():
    # benchmark initualisation
    start = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    
    precision_list = []
    recall_list = []
    fmeasure_list = []
    
    for n in range(5):
        # print(f'---------{n+1}---------')
        df = pd.read_csv(f'Databases/Jul-0{n+1}.csv').iloc[:, 1:]
        df['polyline'] = df.polyline.apply(ast.literal_eval)
        points = []
        
        for polyline in df['polyline']:
            if polyline != []:
                points.append(Point(polyline[0]))
            else:
                points.append(Point((0, 0)))
                
        # R-tree
        gdf = gpd.GeoDataFrame(df, geometry=points)
        within_distance = {}
        visited = set()
        for index0, point in enumerate(points):
            index0_clone = index0
            if point != Point(0, 0): # if not empty coordinates
                gdf_copy = gdf

                # keep checking for points that are within radius and remove them if found.
                while True:
                    indexes = list(gdf_copy.sindex.nearest(point, max_distance=0.0000001)[1])
                    indexes.remove(index0_clone) # remove itself

                    # if no more neighbours, exit loop
                    if len(indexes) == 0:
                        break

                    # add to result
                    if index0 not in within_distance and index0 not in visited:
                        within_distance.update({index0: indexes})
                    elif index0 not in visited:
                        within_distance[index0].append(indexes)

                    visited.update(indexes) # remove duplicates
                 
                    # adjust index after removing row(s)
                    total_diff = 0
                    for idx in indexes:
                        if index0 > idx:
                            total_diff += 1
                    index0_clone = index0_clone - total_diff

                    gdf_copy = gdf_copy.drop(indexes)  # remove the shortest distances

        df_true_values = pd.read_csv(f'Start_points/true_values{n+1}.csv')
        true_values = df_true_values.groupby('point_a_id')['point_b_id'].apply(list).to_dict()

        precision, recall, fmeasure = compare_lists(true_values, within_distance)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)

    end = time.time()
    print('\nR-tree stats')
    print('Avg precison:', np.mean(precision_list))
    print('Avg recall:', np.mean(recall_list))
    print('Avg f-measure:', np.mean(fmeasure_list))
    print('Time taken:', end - start, 'sec')
    print("CPU Usage: {}%".format(cpu_usage))
    print("Memory Usage: {} bytes".format(memory_usage.used))
    print("Read I/Os: {}".format(io_counters.read_count))
    print("Write I/Os: {}".format(io_counters.write_count))

# q5a()
# q5b()