# lstm-jazz
Generate jazz music using an LSTM neural network. [See accompanying blog post.](https://www.rileynwong.com/blog/2019/2/25/generating-music-with-an-lstm-neural-network)

### Setup
```
$ pip install music21
$ pip install keras
$ pip install numpy 
```

### Usage
1. Set training data: Create a folder containing midi files, or use one of the ones provided. `ff_midi_songs` contains music from Final Fantasy, `herbie_midi_songs` contains music by Herbie Hancock. 
2. In `train.py`, line 20: `for f in glob.glob('herbie_midi_songs/*.mid'):`, edit the folder to the folder containing your training set of midi files.
3. Run `$ python train.py`
4. Once trained, you'll have a lot of resulting weight files, in the format `weights-improvement-***.hdf5`. The most recent one will be the one you want to use.
5. In `generate.py`, line 67: `model.load_weights('jazz_weights.hdf5')`, set the path to the weights file.
6. Run `$ python generate.py`. Your resulting song will be created in the same folder as `output_song.mid`

### Credits
https://github.com/Skuldur/Classical-Piano-Composer 
