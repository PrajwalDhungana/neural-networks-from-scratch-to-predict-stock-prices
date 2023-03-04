from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import sys
import yfinance as yf
# to import from a parent directory
sys.path.append('../')
from NeuralNetworks.FeedForward import FeedForward
from NeuralNetworks.rnn_v2 import RNN_V2
from NeuralNetworks.lstm import LSTM
from NeuralNetworks.RNN import RNN
from utils import *
from normalize import Normalize, MinMax

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# prevent caching so website can be updated dynamically
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def lstm_predict(stock, start, end):
    # shift start date -4 days for correct test/train i/o

    # get stock data
    try:
        df, data = get_stock_data(stock, start, end, json=False)
    except:
        # error info
        e = sys.exc_info()
        print(e)
        print("lstm predict fail")
        return e

    stock = df
    data = data.reset_index()
    trend_dates = pd.to_datetime(data["Date"]).map(lambda x: str(x.date())).tolist()
    
    scaler = Normalize(df)
    df = scaler.normalize_data(df)

    train_max_index = round((len(df) - 1) * 0.80)

    training_input_1 = [[df[i-6], df[i-5]] for i in range(6, train_max_index)]
    training_input_2 = [[df[i-4], df[i-3]] for i in range(6, train_max_index)]
    training_input_3 = [[df[i-2], df[i-1]] for i in range(6, train_max_index)]
    target = [[i] for i in df[6:train_max_index]]

    training_input_1 = np.array(training_input_1, dtype=float)
    training_input_2 = np.array(training_input_2, dtype=float)
    training_input_3 = np.array(training_input_3, dtype=float)
    target = np.array(target, dtype=float)

    assert len(training_input_1) == len(training_input_2) == len(training_input_3) == len(target)

    # create neural network
    NN = LSTM()

    # number of training cycles
    training_cycles = 10

    # train the neural network
    for cycle in range(training_cycles):
        for n in training_input_1:
            output = NN.train(training_input_1, training_input_2, training_input_3, target)


    # de-Normalize
    output = scaler.denormalize_data(output)
    target = scaler.denormalize_data(target)

    # transpose
    output = output.T

    # change data type so it can be plotted
    prices = pd.DataFrame(output)

    # [price 2 days ago, price yesterday] for each day in range
    testing_input_1 = [[df[i-6], df[i-5]] for i in range(train_max_index, len(df))]
    testing_input_2 = [[df[i-4], df[i-3]] for i in range(train_max_index, len(df))]
    testing_input_3 = [[df[i-2], df[i-1]] for i in range(train_max_index, len(df))]
    test_target = [[i] for i in df[train_max_index:len(df)]]

    assert len(testing_input_1) == len(testing_input_2) == len(testing_input_3) == len(test_target)

    testing_input_1 = np.array(testing_input_1, dtype=float)
    testing_input_2 = np.array(testing_input_2, dtype=float)
    testing_input_3 = np.array(testing_input_3, dtype=float)
    test_target = np.array(test_target, dtype=float)

    # test the network with unseen data
    test = NN.test(testing_input_1, testing_input_2, testing_input_3)

    # de-Normalize data
    test = scaler.denormalize_data(test)
    test_target = scaler.denormalize_data(test_target)

    # transplose test results
    test = test.T

    # accuracy
    accuracy = 100 - mape(test_target, test)
    print("MSE : ", mse(test_target, test))
    print("RMSE : ", rmse(test_target, test))

    print("DF TEST", pd.DataFrame(test))

    return stock, trend_dates, prices, pd.DataFrame(test), str(round(accuracy, 2))

def handle_nn(stock, start, end):
    return lstm_predict(stock, start, end)

def get_stock_data(ticker, start=[2020, 1, 1], end=[2023, 1, 1], json=True):
    # *list passes the values in list as parameters
    start = dt.datetime(*start)
    end = dt.datetime(*end)

    # download csv from yahoo finance
    try:
        data = yf.download(ticker, start, end)
    except:
        # error info
        e = sys.exc_info()
        print(e)
        print("get data fail")
        return e

    # extract adjusted close column
    df = data["Adj Close"]
    # remove Date column
    df = pd.DataFrame([i for i in df])[0]
    if json:
        # return data as JSON
        return df.to_json()

    else:
        # return data as csv
        return df, data
    
# app routes are urls which facilitate
# data transmit, mainly:
# Get and Post requests
@app.route('/')
def index():
    # load html from templates directory
    return render_template('index.html')

@app.route('/getpythondata')
def get_python_data():
    return get_stock_data("TSLA")

@app.route('/postjsdata', methods=['POST'])
def post_js_data():
    # POST request
    if request.method == 'POST':
        # convert to JSON
        data = request.get_json(force=True)
        stock = data["stock"].upper()
        start = data["startDate"].split("-")
        end = data["endDate"].split("-")

        # convert strings to integers
        start, end = [int(s) for s in start], [int(s) for s in end]

        try:
            # get original stock data, train and test results
            actual, trend_dates, train_res, test_res, accuracy = handle_nn(stock, start, end)
        except:
            # error info
            e = sys.exc_info()
            print(e)
            print("handle_nn fail")
            return "error", 404

        # convert pandas dataframe to list
        actual = [i for i in actual]
        train_res, test_res = [i for i in train_res[0][:]], [i for i in test_res[0][:]]
        train_date, test_date = [i for i in trend_dates[6:len(train_res)]], [i for i in trend_dates[len(train_res)+6:]]

        actualX = trend_dates
        trainX = train_date
        testX = test_date

        # connect training and test lines in plot
        test_res.insert(0, train_res[-1])
        testX.insert(0, trainX[-1])

        return {"stock" : stock,
                "actual" : actual,
                "actualX" : actualX,
                "train" : train_res,
                "trainX" : trainX, 
                "test" : test_res,
                "testX" : testX,
                "accuracy" : accuracy}
        
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 3000, debug=True)