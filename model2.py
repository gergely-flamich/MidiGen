from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import os
import keras.utils as ku
import numpy as np
import json
import argparse

NUM_PITCHES = 128

# velocity, duration, pause
NUM_EXTRA = 0

NUM_INPUTS = NUM_PITCHES + NUM_EXTRA

ROLLBACK = 20

def json_to_list(obj):

    data = np.zeros((NUM_INPUTS,))

    data[obj["pitch"]] = 1
    # data[NUM_PITCHES + 0] = float(obj["vel"])
    # data[NUM_PITCHES + 1] = float(obj["duration"])
    # data[NUM_PITCHES + 2] = float(obj["pause"]) - float(obj["duration"])

    return data

def preprocess(data):

    inputs = [json_to_list(step) for step in data]

    input_ngrams = []
    for i in range(len(inputs)):
        xs = inputs[max(0, i - ROLLBACK + 1):i+1]
        if len(xs) < ROLLBACK:
            diff = ROLLBACK - len(xs)
            pad = [np.zeros((NUM_INPUTS,)) for j in range(diff)]

            xs = pad + xs

        input_ngrams.append(xs)

    X = np.array(input_ngrams[:-1])
    y = np.array(inputs[1:])

    print(X.shape)
    print(y.shape)
    return X,y

def create_model(predictors, label, max_sequence_len, total_words):

    if os.path.exists("midigen.h5"):
        print("Model already exists!")
        return load_model("midigen.h5")

    model = Sequential()
    model.add(LSTM(200, return_sequences = True, input_shape=(ROLLBACK, NUM_INPUTS)))
    #model.add(Dropout(0.2))
    model.add(LSTM(200))
    model.add(Dense(NUM_INPUTS, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=20, verbose=1, callbacks=[earlystop])

    model.save("midigen.h5")
    print(model.summary())
    return model

def generate_text(model, seed_text, next_words, max_sequence_len):

    outputs = []
    test = np.zeros((1, ROLLBACK, NUM_INPUTS))
    test[0, 19, 50] = 1
    # test[0, 19, 128] = 60
    # test[0, 19, 129] = 180
    # test[0, 19, 130] = 180

    for i in range(next_words):

        pred = model.predict(test)

        note = np.argmax(pred[:NUM_PITCHES])

        output = {
            "vel":(int(np.floor(64))),
            "pitch": [int(note)],
            "duration": int(180),
            "pause": int(180)
        }

        outputs.append(output)

        test[0, :19, :] = test[0, 1:, :]
        test[0, 19, :] = pred[0]

        print(test[0, 19, :])

    with open('data.json', 'w') as outfile:
        json.dump(outputs, outfile)


def run(args):
    print(args.data)
    data = json.load(open(args.data))

    predictors, label = preprocess(data)
    # predictors, label, max_sequence_len, total_words = dataset_preparation(data)
    model = create_model(predictors, label, ROLLBACK, NUM_INPUTS)
    print(generate_text(model, "we naughty", 1000, 10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIDI Gen')

    parser.add_argument('--data', type=str, help="Data Path", required=True)

    args = parser.parse_args()

    run(args)
