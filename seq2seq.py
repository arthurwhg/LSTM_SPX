import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, SimpleRNN 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

# Define sequence length
past_days = 5  # Input sequence
future_days = 5  # Output sequence

# Function to create sequences
def create_sequences(data, past_days, future_days):
    X, y = [], []
    for i in range(len(data) - past_days - future_days):
        #date_train.append(data.index[i:i + past_days].to_list())
        X.append(data[i:i+past_days])
        y.append(data[i+past_days:i+past_days+future_days])
    return np.array(X), np.array(y)

# Prepare data sequences
X, y = create_sequences(data_scaled, past_days, future_days)

# Split data into train & test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def LSTMEncoder_Decoder(epochs, batch_size, X_train, y_train, X_test, y_test) :
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
    decoder_outputs = TimeDistributed(Dense(1))(decoder)

    # Define model
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer="adam", loss="mse")

    # Train model
    with tf.device('/gpu:0'):
      history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return history, model


def LSTM(epochs, batch_size, X_train, y_train, X_test, y_test) :
       
    latent_dim = 100

    model = Sequential([
        tf.keras.layers.LSTM(latent_dim, activation="relu", input_shape=(past_days, 1), return_sequences=True),
        tf.keras.layers.LSTM(latent_dim, activation="relu", input_shape=(latent_dim, 1), return_sequences=False),
        Dense(50, activation="relu"),
        Dense(25, activation="relu"),
        Dense(future_days) #
    ])
    
    model.compile(optimizer="adam", loss="mse")
    # Train model
    with tf.device('/gpu:0'):
      history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return history, model
 
def rnn(epochs, batch_size, X_train, y_train, X_test, y_test) :
    model = Sequential([
        SimpleRNN(100, activation='relu', input_shape=(past_days, 1), return_sequences=False),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(future_days)  # This outputs a vector of length future_days
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history, model
    

def calculate_mae(y_acc, y_pred) :
    return np.mean((np.abs(y_acc - y_pred)))

def calculate_mse(y_acc, y_pred) :
    return np.mean((np.square(y_acc - y_pred)))


##### main function starts here
EPOCHS=100
BATCH_SIZE=64
SetModel = 'LSTM' if past_days == future_days else 'LSTMEncoder_Decoder'
history = None
model = None

if SetModel == 'LSTMEncoder_Decoder' :
    history, model = LSTMEncoder_Decoder(EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test)
elif SetModel == 'LSTM' :
    history, model = LSTM(EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test)
elif SetModel == 'RNN' :
    history, model = rnn(EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test)
    

model.summary()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

X_all = np.concatenate((X_train, X_test), axis=0)
print(f"X_all: {X_all.shape}")
print(f"Date: {data['Date'].shape}")

## make dates for all predictions 
dates = data['Date'][:-past_days]
print(f"dates: {dates.shape}")
print(f"last date: {data['Date'].iloc[-1]}")
lastDate = dates.iloc[-1]
## shift dates by 5 days right
last_days = pd.date_range(start=lastDate+pd.Timedelta(days=1), periods=future_days, freq='D')
last_days = pd.Series(last_days)
#dates = pd.concat([pd.Series([None] * past_days), dates.iloc[:-past_days]]).reset_index(drop=True)
#dates = pd.concat((dates, np.array(last_days)), axis=0)
dates = pd.concat([dates, last_days], ignore_index=True)
print(f"dates: {dates.shape}")

mae = 0
mse = 0
# lastPred= y_pred_rescaled[-1]
# firsPred= y_pred_rescaled[0]

# # shift to right by past_days
# #y_pred_rescaled = pd.concat([pd.Series([None] * past_days), y_pred_rescaled.iloc[:-past_days]]).reset_index(drop=True)

# #y_pred_rescaled = np.concatenate(y_pred_rescaled, np.full((past_days,1),lastPred))
# for i in range(past_days):
#     y_pred_rescaled = np.concatenate((firsPred, y_pred_rescaled, lastPred), axis = 0)

# print(f"y_pred_rescaled: {y_pred_rescaled.shape}")

# the final predicted data to be shown in chart
y_pred_final = None
if SetModel == 'LSTMEncoder_Decoder' :
    # Predict future all date
    y_pred = model.predict(X_all)
    y_shift = [y_pred[0] for i in range(past_days)]
    y_pred = np.concatenate((y_shift, y_pred), axis = 0)

    # remove the first past_days no predicition 
    y_original = data_scaled[past_days:]
    # remove the last future_days, no prediction
    #y_original = y_original[:-future_days]
    #mse = calculate_mse(y_original, y_pred[:,0])
    #print(f"MSE: {mse}")

    # # Inverse transform predictions. LSTMEncoder_Decoder model returns array in shape [x,5,1]
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

    print(f"last {future_days} days prediction: {y_pred_rescaled[-1:]}")  
    pred_last_days = []
    pred_Last_list = y_pred_rescaled[-1:]

    # reshape the last futurdays from the last recods and add to pred_rescaled array to display
    for i in range(future_days):
        pred_day = [[pred_Last_list[0,i,0]]]
        for j in range(future_days-1):
            pred_day.append([0])
        pred_last_days.append(pred_day)        
    #print(f"last 5 day prediction: {pred_last_days}") 
    #print(f"last_5_days: {np.array(pred_last_days).shape}")
    y_pred_rescaled = np.concatenate((y_pred_rescaled, np.array(pred_last_days)), axis = 0)
    
    #print(f"y_pred_rescaled: {y_pred_rescaled.shape}")
    y_pred_final = y_pred_rescaled[:,0]
    # calculate actual error
    print(f"pred_final: {y_pred_final.flatten().shape}")
    print(f"original: {data['Close'].values.flatten().shape}")
    mse = calculate_mse(np.array(data['Close'].values.flatten()), np.array(y_pred_final.flatten()))
    mae = calculate_mae(np.array(data['Close'].values.flatten()), np.array(y_pred_final.flatten()))
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    


elif SetModel == 'LSTM' or SetModel == 'RNN' :  
    # the LSTM model returns prediction in shape [x,future_days]
    y_pred = model.predict(X_all)
    y_shift = [y_pred[0] for i in range(past_days)]
    y_pred = np.concatenate((y_shift, y_pred), axis = 0)

    # remove the first past_days no predicition 
    y_original = data_scaled[past_days:]
    # remove the last future_days, no prediction
    y_original = y_original[:-future_days]
    # calculate error using nomalized data
    #mse = calculate_mse(y_original, y_pred[:,0])
    #print(f"MSE: {mse}")

    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

    # reshape the last future days to a an array (future_days, future_days)
    pred_last = y_pred_rescaled[-1:]
    pred_last = pred_last.reshape(future_days, 1)
    # add future_days -1 columns in the array 
    appandix = np.zeros(future_days).reshape(future_days, 1)
    for i in range(future_days-1):
        pred_last = np.concatenate((pred_last, appandix), axis = 1)
    
    #print(f"last 5 day prediction: {pred_last.shape}")
    #print(f"pred_last: {y_pred_rescaled.shape}")

    y_pred_rescaled = np.concatenate((y_pred_rescaled, pred_last), axis = 0)    
    y_pred_final = y_pred_rescaled[:,0]
    # calculate error using original data
    mse = calculate_mse(data['Close'], y_pred_final)
    mae = calculate_mae(data['Close'], y_pred_final)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")


#y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Plot actual vs predicted prices
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Close'], label="Actual Price")
plt.plot(dates, y_pred_final, linestyle="-", label="Predicted Price")
#plt.plot(dates, data_scaled, label="Predicted Price (Next Day)")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price($)")
plt.legend()
plt.title(f"SPX Price Prediction (Predicated by next {future_days} Days Model: {SetModel}, mae {mae:.2f}$)")
plt.grid(visible=True, axis='both')
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
