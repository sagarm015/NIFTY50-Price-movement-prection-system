import quandl
import os
import pandas as pd
import pickle
import bs4 as bs
import requests
from sklearn import preprocessing
import warnings
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical

#os.chdir('/Users/divyanshbhadauria/Desktop/SESE')
os.chdir('S:\semester 4\SE\SEML\SEML')
quandl.ApiConfig.api_key = 'ShYW25mfgL56v8tkEJ61'
startdate = "2012-11-22"
enddate = "2018-11-22"


def nifty_50_list():
    resp = requests.get('https://en.wikipedia.org/wiki/NIFTY_50')
    soup = bs.BeautifulSoup(resp.text, 'lxml')

    table = soup.find('table', {'class': 'wikitable sortable'}, 'tbody')

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker.split('.')[0])

    with open("nifty50_list.pickle", "wb") as f:
        pickle.dump(tickers, f)

    tickers = list(map(lambda x: x.replace("BAJAJ-AUTO", "BAJAJ_AUTO"), tickers))
    tickers = list(map(lambda x: x.replace("M&M", "MM"), tickers))
    tickers.append('NIFTY_50')
    tickers.remove('UPL')
    return tickers


def get_nifty50_list(scrap=False):
    if scrap:
        tickers = nifty_50_list()
    else:
        with open("nifty50_list.pickle", "rb") as f:
            tickers = pickle.load(f)
    return tickers



def getStockdataFromQuandl(ticker):
    quandl_code = "NSE/" + ticker
    try:
        if not os.path.exists(f'stock_data/{ticker}.csv'):
            data = quandl.get(quandl_code, start_date=startdate, end_date=enddate)
            data.to_csv(f"S:\semester 4\SE\SEML\SEML/stock_data/{ticker}.csv")
            #data.to_csv(f"/Users/divyanshbhadauria/Desktop/SESE/stock_data/{ticker}.csv")
        else:
            print(f"stock data for {ticker} already exists")
    except quandl.errors.quandl_error.NotFoundError as e:
        print(ticker)
        print(str(e))


def load():
    tickers = get_nifty50_list(True)
    df = pd.DataFrame()
    for ticker in tickers:
        getStockdataFromQuandl(ticker)
        try:
            data = pd.read_csv(f'stock_data/{ticker}.csv')

            if ticker == "NIFTY_50":
                data.rename(columns={'Close': f"{ticker}_Close", 'Shares Traded': f"{ticker}_Volume"}, inplace=True)
            else:
                data.rename(columns={'Close': f"{ticker}_Close", 'Total Trade Quantity': f"{ticker}_Volume"},
                            inplace=True)

            df = pd.concat([df, data[f'{ticker}_Volume'], data[f'{ticker}_Close']], axis=1)
        except Exception as e:
            print(f"couldn't find {ticker}")
            print(str(e))
    df.to_csv('nifty50_closingprices.csv')
    df.dropna(inplace=True)
    return df



SERIES_LENGTH = 30
PREDICT_LENGTH = 7

TICKER = "NIFTY_50"


def normalize_data(df):
    pass


def scale_data(df):
    for column in df.columns:
        df[column] = preprocessing.scale(df[column].values)
    return df


def process_data(df):
    df["nifty_future_price"] = df[f"{TICKER}_Close"].shift(-PREDICT_LENGTH)  # Shift it by 7 days

    # Dropping any Nan values
    df.dropna(inplace=True)

    df["Label"] = np.where(df["nifty_future_price"] >= df["NIFTY_50_Close"], 1, 0)
    df.drop('nifty_future_price', 1, inplace=True)
    df.to_csv('nifty50_future_label.csv')

    sequence = []
    temp = df.loc[:, df.columns != 'Label']
    temp = scale_data(temp)
    for i in range(len(temp) - SERIES_LENGTH):
        sequence.append([np.array(temp[i:i + SERIES_LENGTH]),
                         df.iloc[i + SERIES_LENGTH, -1]])

    np.random.shuffle(sequence)

    X = []
    y = []
    buy = []
    sell = []
    for seq, label in sequence:
        if label == 0:
            sell.append([seq, label])
        else:
            buy.append([seq, label])


    buys = len(buy)
    sells = len(sell)
    print(f"original buys:{buys} original sells:{sells}")
    if buys < sells:
        buy = buy[:buys]
        sell = sell[:buys]
    else:
        buy = buy[:sells]
        sell = sell[:sells]

    print(f"buys:{len(buy)} sells:{len(sell)}")
    # Concat the buys an sells and shuffle it out again
    sequence = buy + sell

    np.random.shuffle(sequence)

    for seq, label in sequence:
        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)

# this is my part

df = load()
process_data(df)

df = load()
training_size = 0.8
spilt_point = int(training_size * len(df))

train_df = df[:spilt_point]
test_df = df[spilt_point:]

print(f"train_df {train_df[:5]}")
print("=" * 100)
print(f"test_df {test_df[:5]}")

warnings.filterwarnings("ignore")
train_x, train_y = process_data(train_df)
test_x, test_y = process_data(test_df)

print('X_train :', train_x.shape)
print('y_train :', train_y.shape)
print('X_test :', test_x.shape)
print('y_test :', test_y.shape)

NAME = "NIFTY50PRED"
BATCH_SIZE = 64
EPOCHS = 15

to_categorical([0, 1])



def build_model():
    model = Sequential()
    model.add(LSTM(256, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y)
    print("Validation accuracy percentage", score[1] * 100)
    print("Validation loss percentage", score[0] * 100)
    return model


model = build_model()
prediction = model.predict(test_x)
plt.plot(prediction[40:50], color='green', label='predicted_data')
plt.plot(test_y[40:50], color='blue', label='actual_data')
plt.interactive(False)

plt.show()

