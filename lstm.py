import glob
import pickle
from music21 import converter, instrument, note, chord


def get_notes():
    """
    Convert midi songs to notes. Serialize when done.
    """

    notes = []

    for f in glob.glob('midi_songs/*.mid'):
        print('Parsing song: ', f)
        midi = converter.parse(f)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: # if file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # notes are flat stucture
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def load_notes():
    """
    Deserialize notes file.
    """
    notes = []

    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    return notes
