import os
import urllib.request
import json
from datetime import datetime
import pandas as pd
import numpy as np

def race_results_download(file, url):
    race_results = []

    if not os.path.isfile(file): urllib.request.urlretrieve(url, file)

    with open(file, 'rt') as json_data:
        load = (json.loads(json_data.read()))

    for race in load['MRData']['RaceTable']['Races']:
        for result in race['Results']:
            race_results.append({"season": int(race['season']),
                                 "round": int(race['round']),
                                 "circuit": race['Circuit']['circuitId'],
                                 "driver": result['Driver']['driverId'],
                                 "age": driver_age_rounded(result['Driver']['dateOfBirth'], race['date']),
                                 "constructor": result['Constructor']['constructorId'],
                                 'grid': int(result['grid']),
                                 "position": int(result['position'])})

    return race_results

def race_results_year(year):
    file = f"race_results_{year}.json"
    url = f"http://ergast.com/api/f1/{year}/results.json?limit=600"
    return race_results_download(file, url)

def race_results_year_round(year, round):
    file = f"race_results_{year}_{round}.json"
    url = f"http://ergast.com/api/f1/{year}/{round}/results.json"
    return race_results_download(file, url)

def driver_standings_download(year, round):
    driver_standings = []

    file = f"standings/driver_standings_{year}_{round}.json"
    url = f"http://ergast.com/api/f1/{year}/{round}/driverStandings.json"

    if not os.path.isfile(file): urllib.request.urlretrieve(url, file)

    with open(file, 'rt') as json_data:
        load = (json.loads(json_data.read()))

    for standingsList in load['MRData']['StandingsTable']['StandingsLists']:
        for standing in standingsList['DriverStandings']:
            driver_standings.append({"season": int(standingsList['season']),
                                     "round": int(standingsList['round']),
                                     "driver": standing['Driver']['driverId'],
                                     "standing": int(standing['position'])})

    return driver_standings

# def driver_age(date_of_birth, date_of_race):
#     start=datetime.strptime(date_of_birth, '%Y-%m-%d')
#     end=datetime.strptime(date_of_race, '%Y-%m-%d')
#     delta = relativedelta.relativedelta(end, start)
#     return delta.years

def driver_age_rounded(date_of_birth, date_of_race):
    start=datetime.strptime(date_of_birth, '%Y-%m-%d').year
    end=datetime.strptime(date_of_race, '%Y-%m-%d').year
    return end-start

def number_of_races(dataset):
    max_races={}
    for data in dataset:
        races=max_races.get(data['season'], 0)
        if data['round']>races: max_races['season']=data['round']
    return max_races

def add_last_position(dataset):
    drivers_last_race={}
    last_race=race_results_year_round(dataset[0]['season']-1, "last")

    for data in last_race:
        if data['position']>20: drivers_last_race[data['driver']]=20
        else: drivers_last_race[data['driver']]=data['position']

    for data in dataset:
        last=drivers_last_race.get(data['driver'], data['grid'])
        data['last_result']=last
        drivers_last_race[data['driver']]=data['position']

    return dataset

def add_standings(dataset):
    drivers_last_standing={}
    season=dataset[0]['season']
    round=dataset[0]['round']
    last_standings=driver_standings_download(dataset[0]['season']-1, "last")

    for standing in last_standings:
        drivers_last_standing[standing['driver']]=standing['standing']

    for data in dataset:
        if data['round']==round:
            last = drivers_last_standing.get(data['driver'], data['grid'])
            data['standing']=last
        else:
            last_standings=driver_standings_download(season, round)
            for standing in last_standings:
                drivers_last_standing[standing['driver']] = standing['standing']
            season=data['season']
            round=data['round']
            last = drivers_last_standing.get(data['driver'], data['grid'])
            data['standing'] = last

    return dataset

def download_in_range(start, end):
    dataset=[]
    for i in range(end-start+1):
        dataset+=race_results_year(start+i)
    return dataset

def dict_to_dataframe(dataset):
    dict={"season": [], "round": [], "circuit": [], "driver": [], "age": [], "constructor": [], "standing": [],
          "last_result": [], "grid": [], "position": []}
    for data in dataset:
        dict['season'].append(data['season'])
        dict['round'].append(data['round'])
        dict['circuit'].append(data['circuit'])
        dict['driver'].append(data['driver'])
        dict['age'].append(data['age'])
        dict['constructor'].append(data['constructor'])
        dict['standing'].append(data['standing'])
        dict['last_result'].append(data['last_result'])
        dict['grid'].append(data['grid'])
        dict['position'].append(data['position'])
    return pd.DataFrame.from_dict(dict)

def median_position_for_age(dataset, ages):
    y=np.array([])
    for age in ages:
        positions=np.array([])
        for i in range(len(dataset)):
            if dataset['age'].values[i]==age: positions=np.append(positions, dataset['position'].values[i])
        y=np.append(y, np.median(positions))
    return y

def team_name_fixup(dataset):
    for i in range(len(dataset)):
        if dataset['constructor'].values[i] == 'renault':
            dataset['constructor'].values[i] = 'alpine'
        elif dataset['constructor'].values[i] == 'force_india':
            dataset['constructor'].values[i] = 'aston_martin'
        elif dataset['constructor'].values[i] == 'toro_rosso':
            dataset['constructor'].values[i] = 'rb'
        elif dataset['constructor'].values[i] == 'alphatauri':
            dataset['constructor'].values[i] = 'rb'
        elif dataset['constructor'].values[i] == 'alfa':
            dataset['constructor'].values[i] = 'sauber'
        elif dataset['constructor'].values[i] == 'racing_point':
            dataset['constructor'].values[i] = 'aston_martin'
    return dataset

