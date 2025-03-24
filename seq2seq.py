import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, SimpleRNN,Dropout
 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Check if GPU is available
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Is GPU available?", tf.test.is_gpu_available())

# set to use GPU on mac
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load SPX data (Replace with actual file path)
data = pd.read_csv("./data/spx2010.csv",encoding="ISO-8859-1")  # Assume CSV has 'Date' and 'Close' columns
data['Date'] = pd.to_datetime(data['Date'])
# sort data by date
data.sort_values('Date',inplace=True) 
print(f"data: {data.shape}")

print(data.head())
print(data.head(-5))

plt.figure(figsize=(20,5))
plt.plot(data['Date'],data['Close'])
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price($)")
plt.title("SPX Price between 2010/1-2025/3")
plt.grid(visible=True, axis='both')
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Normalize price data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close']])
print(f"data_scaled: {data_scaled.shape}")

#

# Define sequence length
past_days = 5  # Input sequence
future_days = 5  # Output sequence

# Function to create sequences for X y, dates
def create_sequences(data, date, past_days, future_days):
    X, y, dateList = [], [], []
    quotient, mod = divmod(len(data), past_days)
    sections = quotient if mod > 0 else quotient -1 
    for i in range(sections):
        #date_train.append(data.index[i:i + past_days].to_list())
        start = i * past_days
        end = start + past_days + future_days
        X.append(data[start:start+past_days])
        # ignore the last group target (not sufficent features to predict)
        if end < len(data) - mod :
            y.append(data[start+past_days:end])
        dateList.append(date[start:start+past_days])
    
    xArray = np.array(X)
    yArray = np.array(y)
    dateArray = np.array(dateList)
    return xArray, yArray, dateArray

# Prepare data sequences
# DateList: all dates to be predicted
# X: features [past_days,1]
# y: labels [len(DateList),1]
#X, y, DateList = create_sequences(data_scaled, data['Date'], past_days, future_days)

# # Split data into train & test sets
# split = int(len(X) * 0.8)
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

###
# create a date list for all predicted data for plot 
def createDates(date_list, original_dates, past_days,future_days) :
    quotient, mod = divmod(len(original_dates), past_days)
    dates = original_dates[0:quotient * past_days].to_numpy()
    lastDates = dates[-mod:]
    dates = np.append(dates,lastDates)

    #lastDate = dates[len(dates)-1]
    #print(f"total: {len(dates)} days +")
    # if mod > 0:
    #     # all dates being predicted
    #     for i in range(0,mod):
    #         dates = np.append(dates,original_dates[i-mod-1])
            # skip Saturday and Sunday
            # date_last = lastDate + np.timedelta64(i, 'D')
            # print(f"{date_last}: {date_last.weekday()}")
            # if pd.Timestamp((lastDate + np.timedelta64(i, 'D'))).weekday() not in [6,7]:
            #     lastdate_t = pd.Timestamp(lastDate + np.timedelta64(i, 'D')).to_pydatetime()
            #     dates= np.append(dates,lastdate_t)
            # else:
            #     print(f"ignored: {date_last}")

    # length is length of date_list + future_days
    return dates
 
###
# model of LSTM on EncoderDecoder 
def LSTMEncoder_Decoder(epochs, batch_size, X_train, y_train, X_test, y_test, past_days, future_days) :
    # LSTM Encoder-Decoder Model
    latent_dim = 50

    # Encoder
    encoder_inputs = Input(shape=(past_days, 1))
    encoder = tf.keras.layers.LSTM(latent_dim, activation="relu", return_sequences=True)(encoder_inputs)
    encoder = tf.keras.layers.LSTM(latent_dim, activation="relu", return_sequences=False)(encoder_inputs)
    encoded = RepeatVector(future_days)(encoder)

    # Decoder
    decoder = tf.keras.layers.LSTM(latent_dim, activation="relu", return_sequences=True)(encoded)
    decoder = tf.keras.layers.LSTM(latent_dim, activation="relu", return_sequences=True)(encoded)
    decoder_outputs = TimeDistributed(Dense(future_days))(decoder)

    # Define model
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer="adam", loss="mse",metrics=['accuracy'])

    # Train model
    with tf.device('/gpu:0'):
      history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return history, model


###
# model of LSTM on LSTM
def LSTM(epochs, batch_size, X_train, y_train, X_test, y_test, past_days, future_days) :
       
    latent_dim = 100

    model = Sequential([
        tf.keras.layers.LSTM(latent_dim, activation="relu", input_shape=(past_days, 1), return_sequences=True),
        tf.keras.layers.LSTM(latent_dim, activation="relu", input_shape=(latent_dim, 1), return_sequences=True),
        Dense(50, activation="relu"),
        Dense(25, activation="relu"),
        Dense(future_days) #
    ])
    
    model.compile(optimizer="adam", loss="mse",metrics=['accuracy'])
    # Train model
    with tf.device('/gpu:0'):
      history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return history, model
 
###
# model of RNN
def rnn(epochs, batch_size, X_train, y_train, X_test, y_test, past_days, future_days) :
    model = Sequential([
        SimpleRNN(100, activation='relu', input_shape=(past_days, 1), return_sequences=False),
        Dense(50, activation='relu'),
        Dense(future_days)  # This outputs a vector of length future_days
    ])
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    # Train model
    with tf.device('/gpu:0'):
       history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    return history, model
    

def calculate_mae(y_acc, y_pred) :
    return np.mean((np.abs(y_acc - y_pred)))

def calculate_mse(y_acc, y_pred) :
    return np.mean((np.square(y_acc - y_pred)))


###
# model of lstm stack on RNN
def stack(epochs, batch_size, X_train, y_train, X_test, y_test, past_days, future_days):
       
    latent_dim = 60

    model = Sequential([
        tf.keras.layers.LSTM(latent_dim, activation="relu", input_shape=(past_days, 1), return_sequences=True),
        SimpleRNN(latent_dim, activation='relu', input_shape=(latent_dim, 1), return_sequences=False),
        Dense(25, activation="relu"),
        #Dense(10, activation="relu"),
        Dense(future_days) #
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    # Train model
    with tf.device('/gpu:0'):
      history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return history, model


###
# show loss and accuracy plot 
def showLossandAccuracyPlot(history):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')

    ax[1].set_title('Training and Validation Accuracy')
    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.legend()
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()

###
# use all sample data create a full list of X and y for prediction
def create_full_sequences(data, past_days, future_days):
    # X all list, y: the first past_days cannot be preded, use the sample data
    X = [[[0] for _ in range(past_days)] for _ in range(past_days)]
    y = [] 

    # initiate y by the fist past_days of sequence [past_days, future_days, 1] 
    for i in range(past_days):
        f = data[i+past_days:i+past_days+future_days]
        y.append(f)
        
    for i in range(len(data) - past_days - future_days):
        X.append(data[i:i+past_days].tolist())
        y.append(data[i+past_days:i+past_days+future_days].tolist())

    ##
    # length x = length of data - future_days (the first past_days are 0, do not need to predict)
    # length y = length of data - future_days (ths ifrst past_days are same value, do not need to predict)
    return np.array(X), np.array(y)
    
   

###
# build up the predicted arrry from predicted result with process
# update first past_days by the original value (cannot be predicted)
# expend the last future_days result from the last predicted data
def buildPredictedArray(y_predicted, y_original, past_days, future_days):
    y_last = [] 
    quotient, mod = divmod(len(y_original), past_days)
    
    for i in range(past_days):
        y_last.append(y_original[i][0])
        
    # for i in range(len(y_predicted)-past_days):
    #     y_last.append(y_predicted[i+past_days][0])

    # add each prediction into the last list (each prediction is for futur_days)
    for i in range(1,len(y_predicted)) :
        for pred in y_predicted[i] :
            y_last.append(pred)            

    # add future days
    # lastPreds = y_predicted[-1]
    # for i in range(future_days):
    #     dayi = lastPreds[i]
    #     y_last.append(dayi)

    # add remaining days
    lastValues = y_original[-mod:]
    y_last = np.append(y_last, lastValues)

    #)
    # for i in range(mod):
    #     y_last.append(y_original[i-mod][0])

    # length of y_last = past_days + len(y_predicted)*past_days - past_days + future_days + mod days 
   #         = len(y_predicted)*past_days + future_days + mod days
    return np.array(y_last)


###
#
def showPredictionandAccutual(dates, y_pred_final, date_actual, y_actual, mae, mse) :

    #x_numeric = mdates.date2num(dates)

    # Plot actual vs predicted prices
    plt.figure(figsize=(16,9))
    plt.plot(dates, y_actual, label="Actual Price")
    plt.plot(dates, y_pred_final, linestyle="-", label="Predicted Price")
    #plt.plot(dates, data_scaled, label="Predicted Price (Next Day)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Price($)")
    plt.title(f"SPX Price Prediction (Predicated by next {future_days} Days Model: {SetModel}, mae {mae:.2f}$)")
    plt.grid(visible=True, axis='both')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.legend()
    plt.show()


###
# transform the predction to inverse scaled
def inverseTransform(y_pred, scaler):
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    #y_pred_rescaled = scaler.inverse_transform(y_pred)
    return y_pred_rescaled


def showData(actual, predicted, dates):

    print(f"#Date\tActual\tPredicted\tgap")
    for i in range(len(predicted)):
        showd = pd.Timestamp(dates[i])
        print(f'{i}\t{showd.strftime("%Y-%m-%d")}\t{actual[i]}\t{predicted[i]:.2f}\t{predicted[i]-actual[i]:.2f}')

    print("====end===")

def showDataArray(X, y, x_original, y_original, dates, dates_original):
    index = 0
    for i in range(len(dates)-1):
        for j in range(past_days):
            showd = pd.Timestamp(dates[index])
            showd2 = pd.Timestamp(dates_original[index])
            print(f'{i}\t{showd.strftime("%Y-%m-%d")}\t{showd2.strftime("%Y-%m-%d")}\t{X[i][j]}\t{x_original[index]}\t{y[i][j]}\t{y_original[index+future_days]}')
            index+=1

##### main function starts here
EPOCHS=60
BATCH_SIZE=32
SetModel = 'STACK' #if past_days == future_days else 'LSTMEncoder_Decoder'
history = None
model = None
mae = 0
mse = 0
y_pred_final = None

# Prepare data sequences for training
# DateList: all dates to be predicted
# X: features [past_days,1]
# y: labels [len(DateList),1]
X, y, DateList = create_sequences(data_scaled, data['Date'], past_days, future_days)
# Split data into train & test sets
split = int(min(len(X),len(y)) * 0.8)
len_test = len(y) if len(y) < len(X) else len(x)
X_train, X_test = X[:split], X[split:len_test]
y_train, y_test = y[:split], y[split:len_test]

#X_all = np.concatenate((X_train, X_test), axis=0)
#X_all, y_all, date_all = create_sequences(data_scaled, data['Date'], past_days, future_days)
print(f"X_all: {X.shape}")
print(f"y_all: {y.shape}")
print(f"date_all_groups: {DateList.shape}")
#print(f"Dates grouped to be predicted: {len(DateList)}")
dates = createDates(DateList, data['Date'], past_days, future_days)
print(f"dates to be predicted: {np.array(dates).shape}")

if SetModel == 'LSTMEncoder_Decoder' :
    history, model = LSTMEncoder_Decoder(EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test, past_days, future_days)
    y_pred = model.predict(X)
    print(f"y_pred: {y_pred.shape}")
    y_pred_final = buildPredictedArray(y_pred, data_scaled, past_days, future_days)
    print(f"y_pred_final: {y_pred_final.shape}")
    mae = calculate_mae(data['Close'], y_pred_final)
    mse = calculate_mse(data['Close'], y_pred_final)
    y_pred_rescaled = inverseTransform(y_pred, scaler)
    showLossandAccuracyPlot(history)
    showPredictionandAccutual(dates, y_pred_rescaled, data['Date'], data['Close'], mae, mse)     

elif SetModel == 'LSTM' :
    history, model = LSTM(EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test, past_days, future_days)
    y_pred = model.predict(X)
    y_pred_final = buildPredictedArray(y_pred, data_scaled, past_days, future_days)
    mae = calculate_mae(data['Close'], y_pred_final)
    mse = calculate_mse(data['Close'], y_pred_final)
    print(f"mae: {mae} mse:{mse}")
    y_pred_rescaled = inverseTransform(y_pred_final, scaler)
    showLossandAccuracyPlot(history)
    model.summary()
    #showData(data['Close'], y_pred_rescaled, dates)
    #showDataArray(X, y, data_scaled, data_scaled, dates, data['Date'])
    showPredictionandAccutual(dates, y_pred_rescaled, data['Date'], data['Close'], mae, mse)     

elif SetModel == 'RNN' :
    history, model = rnn(EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test, past_days, future_days)
    y_pred = model.predict(X)
    y_pred_final = buildPredictedArray(y_pred, data_scaled, past_days, future_days)
    mae = calculate_mae(data['Close'], y_pred_final)
    mse = calculate_mse(data['Close'], y_pred_final)
    y_pred_rescaled = inverseTransform(y_pred_final, scaler)
    showLossandAccuracyPlot(history)
    showPredictionandAccutual(dates, y_pred_rescaled, data['Date'], data['Close'], mae, mse)     

elif SetModel == 'STACK' :
    history, model = stack(EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test, past_days, future_days)
    y_pred = model.predict(X)
    print(f"y_pred: {y_pred.shape}")
    y_pred_final = buildPredictedArray(y_pred, data_scaled, past_days, future_days)
    print(f"y_pred_final: {y_pred_final.shape}")
    y_pred_rescaled = inverseTransform(y_pred_final, scaler)
    mae = calculate_mae(data['Close'], y_pred_rescaled)
    mse = calculate_mse(data['Close'], y_pred_rescaled)
    showLossandAccuracyPlot(history)
    model.summary()
    #showData(data['Close'], y_pred_rescaled, dates)
    showPredictionandAccutual(dates, y_pred_rescaled, data['Date'], data['Close'], mae, mse)     
    


