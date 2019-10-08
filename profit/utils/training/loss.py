import keras.backend as K

def std_mae(std=1):
    def mae(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true)) * std
    return mae


def std_rmse(std=1):
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)))) * std
    return rmse


def std_r2(std=1):
    def r2(y_true, y_pred):
        ss_res = K.sum(K.square((y_true - y_pred) * std))
        ss_tot = K.sum(K.square((y_true - K.mean(y_true) * std)))
        return 1 - ss_res / (ss_tot + K.epsilon())
    return r2
    