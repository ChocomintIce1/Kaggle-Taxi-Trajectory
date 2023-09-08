import ast
import pandas as pd
from shapely.geometry import LineString, Point
from shapely import wkt
from geopy.distance import great_circle
import great_circle_calculator.great_circle_calculator as gcc # bearings

df = pd.read_csv('train.csv')

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

# add a lingstring column for postgresql
def linestring():
    '''
    A function that add a linestring column for the database. 
    '''
    df['linestring'] = ''
    for col, polyline in enumerate(df['POLYLINE']):
        if polyline == '':
            pass

        else:
            polyline = ast.literal_eval(polyline)

            if len(polyline) > 1:
                result = 'LINESTRING('
                for x, y in polyline:
                    result += f'{x} {y}, '
                result = result[:-2] + ')'

                df.at[col, 'linestring'] = result

            elif len(polyline) == 1:
                polyline = polyline[0]
                result = f'POINT({polyline[0]} {polyline[1]})'

                df.at[col, 'linestring'] = result

    df.to_csv('train_linestring.csv')

def skyline():
    '''
    A function that preprocesses the data by adding a distance and time column to the database.
    '''
    for i in range(5):
        # df = pd.read_csv('edited/train_filitered_date.csv').iloc[:,:-1] # drop linestring
        df = pd.read_csv(f'Databases/Jul-0{i+1}.csv')
        df['distance'] = ''
        df['time'] = ''

        for col, polyline in enumerate(df['polyline']):
            if polyline == '':
                pass
            
            else:
                polyline = ast.literal_eval(polyline)
                if len(polyline) <= 1: # find unfinished travel
                    pass

                elif LineString(polyline).length == 0: # remove when travelled for 0 distance
                    pass

                else:
                    df.at[col, 'distance'] = great_circle_dist_line(LineString(polyline))
                    df.at[col, 'time'] = (len(polyline) - 1) * 15

        df.to_csv(f'Skyline/train_skyline{i+1}.csv')

def to_string(line: str):
    '''
    Helper function.
    Converts line to a readable string.
    format is: "distance,angle|distance,angle|..."
    '''
    result = ''
    if len(line) == 0: # if empty
        return result
    
    line = ast.literal_eval(line) # convert string to a list
    
    for index, _ in enumerate(line):
        if isinstance(line[0], int): # skip over points
            pass
        
        elif index + 1 < len(line): # if there is next coordinate
            if len(result) != 0: # if first iteration
                result += '|'

            p0 = Point(line[index])
            p1 = Point(line[index + 1])
            
            
            distance = round(great_circle(p0.coords, p1.coords).km, 4)
            angle = round(gcc.bearing_at_p1(list(p0.coords)[0], list(p1.coords)[0]))
            
            result += str(distance) + ',' + str(angle)

    return result

def sim_trajectory():
    '''
    Adds an additional column of the taxi's trip to a readable string
    '''
    for i in range(5):
        # df = pd.read_csv('edited/train_filitered_date.csv').iloc[:,:-1] # drop linestring
        df = pd.read_csv(f'Databases/Jul-0{i+1}.csv')
        df['to_string'] = ''
        df['to_string'] = df['polyline'].apply(to_string)
        df.to_csv(f'Sim_trajectories/sim{i+1}.csv')

def start_points():
    '''
    Adds another column with the taxi's starting position.
    '''
    for i in range(5):
        df = pd.read_csv(f'Databases/Jul-0{i+1}.csv')
        df['polyline'] = df.polyline.apply(ast.literal_eval)
        df['point'] = ''
        
        for col, line in enumerate(df['polyline']):
            if line != []:
                df.at[col, 'point'] = Point(line[0])

        df.to_csv(f'Start_points/train_start{i+1}.csv')

# sim_trajectory()