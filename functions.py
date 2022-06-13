from sklearn import metrics
import requests
from scipy.optimize import curve_fit
import pandas as pd
import cr_model


def get_parameters(min_cr):
    if min_cr >= 1:
        parameters = []
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


def get_data_from_json(min_cr):
    data = pd.read_json('monster_list.json')
    data = filter_data(data, min_cr)
    return data


def chose_cut_point(data, start_index, end_index):
    result = []
    data = data
    for i in range(start_index, end_index):
        learn = data[:i]
        test = data[i:]
        l_x_g, l_y_g, t_x_g, t_y_g = get_x_and_y(learn, test, "cr")

        params, _ = curve_fit(
            cr_model.model_fun, xdata=l_x_g, ydata=l_y_g.values.ravel()
        )
        model = cr_model.CRModel(params)
        y_pred_g = model.run(t_x_g)
        error = round(metrics.mean_absolute_error(t_y_g, y_pred_g))
        result.append((error, i))
    result.sort(key=lambda tup: tup[0])
    return result[0][1]


def get_min_max_of_frame(frame):
    min_col = {}
    max_col = {}
    for col in frame:
        max_col[col] = frame[col].max()
        min_col[col] = frame[col].min()

    result = pd.DataFrame([min_col, max_col], index=['min', 'max'])
    return result


def get_x_and_y(learn, test, y_type):
    learnX = learn[["hp", "ac"]].copy()
    learnY = learn[[y_type]].copy()
    testX = test[["hp", "ac"]].copy()
    testY = test[[y_type]].copy()

    return learnX, learnY, testX, testY
