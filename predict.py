import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation

def generate():
    """ Generate a piano midi file """
    with open('data/notes', 'rb') as f:
        notes = pickle.load(f)

    # Get pitch names
    pitchnames = sorted(set(notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)


def prepare_sequences(notes, pitchnames, n_vocab):
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    # Reshape
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)



def create_network(network_input, n_vocab):
    """ Recreate neural network structure. """
    print('Creating network...')

    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
        ))
    model.add(Dropout(0.3)) # Fraction of input units to be dropped during training
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab)) # Number of possible outputs
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load trained weights
    model.load_weights('weights.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from neural net based on input sequence of notes. """
    print('Generating notes...')

    # Pick random sequence from input as starting point
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # Generate 200 notes
    n = 250
    for note_index in range(n):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        # Take most probable prediction, convert to note, append to output
        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # Scoot input over by 1 note
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output):
    print('Creating midi...')
    """ Convert prediction output to notes. Create midi file!!!! """
    offset = 0
    output_notes = []
    # Possible extension: multiple/different instruments!
    stored_instrument = instrument.Piano()

    # Create Note and Chord objects
    for pattern in prediction_output:
        # Pattern is a Chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = stored_instrument
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else: # Pattern is a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = stored_instrument
            output_notes.append(new_note)

        # Increase offset for note
        # Possible extension: ~ RHYTHM ~
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')


if __name__ == '__main__':
    generate()




