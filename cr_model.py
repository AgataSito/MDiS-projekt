def model_fun(x, a, b, c):
    return a * x["hp"] + b * x["ac"] + c


class CRModel:
    def __init__(self, params):
        self.params = params

    def run(self, x):
        return model_fun(x, *self.params)
