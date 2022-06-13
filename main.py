import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn as sns
from scipy.optimize import curve_fit
import functions
import cr_model

MINCR = 0
MINTRAINING = 10
MINTEST = 6


def run():
    # data = functions.get_data_from_api(MINCR)
    data = functions.get_data_from_json(MINCR)

    cut_point = functions.chose_cut_point(data, MINTRAINING, len(data) - MINTEST)
    print("Cut on: ", cut_point)

    learn = data[:cut_point]
    test = data[cut_point:]

    l_x_g, l_y_g, t_x_g, t_y_g = functions.get_x_and_y(learn, test, "cr")

    params, _ = curve_fit(
        cr_model.model_fun, xdata=l_x_g, ydata=l_y_g.values.ravel()
    )
    model = cr_model.CRModel(params)
    y_pred_g = model.run(t_x_g)

    print("Avg mistake:")
    print(round(metrics.mean_absolute_error(t_y_g, y_pred_g), 2))
    print("Avg mistake%:")
    print(round(metrics.mean_absolute_percentage_error(t_y_g, y_pred_g) * 100, 2), "%")

    test["index"] = test.index
    learn["index"] = learn.index
    listaPred = list(y_pred_g)
    listaTrue = list(test["cr"])
    listaIndex = list(test["index"])
    listaLearn = list(learn["cr"])
    listaLearnIn = list(learn["index"])

    # Wykresy
    sns.scatterplot(x="index", y="cr", data=learn)
    sns.scatterplot(x="index", y="cr", data=test)
    sns.lineplot(x=listaLearnIn, y=listaLearn)
    sns.lineplot(x=listaIndex, y=listaPred, color="red", label='Predicted')
    sns.lineplot(x=listaIndex, y=listaTrue, color="blue", label="Real")
    plt.show()


if __name__ == '__main__':
    run()
