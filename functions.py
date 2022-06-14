import requests
import pandas as pd


def get_parameters(min_cr):
    parameters = []
    if min_cr >= 1:
        for i in range(min_cr, 31):
            parameters.append(i)
    elif 1 > min_cr > 0:
        parameters = get_parameters(1)
        helper = min_cr
        while helper != 1:
            parameters.append(helper)
            helper = helper * 2
    elif min_cr == 0:
        parameters = get_parameters(0.125)
        parameters.append(0)
    return parameters


def get_data_from_api(min_cr):
    parameters = get_parameters(min_cr)
    monster_name = []
    monster_hp = []
    monster_ac = []
    monster_cr = []
    for i in parameters:
        response = requests.get("https://www.dnd5eapi.co/api/monsters", params={"challenge_rating": i})
        monster_list = response.json()['results']
        for each in monster_list:
            monster = requests.get("https://www.dnd5eapi.co/api/monsters/" + str(each['index']))
            monster_name.append(monster.json()['name'])
            monster_hp.append(monster.json()['hit_points'])
            monster_ac.append(monster.json()['armor_class'])
            monster_cr.append(monster.json()['challenge_rating'])
    monsters_dict = {'name': monster_name, 'hp': monster_hp, 'ac': monster_ac, 'cr': monster_cr}
    data = pd.DataFrame.from_dict(monsters_dict)
    return data


def filter_data(data, min_cr):
    data = data[data.cr >= min_cr]
    return data


def get_max_and_min(y):
    helper_min = 30
    helper_max = 0
    for each in y:
        if each > helper_max:
            helper_max = each
        if each < helper_min:
            helper_min = each
    return helper_max, helper_min


def get_data_from_json(min_cr):
    data = pd.read_json('monster_list.json')
    data = filter_data(data, min_cr)
    data = data.reset_index()
    del data['index']
    return data
