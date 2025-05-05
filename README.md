# MusicMOE

Download midi dataset and place them under /data: [rwc pop](https://drive.google.com/file/d/1xeWFc_cfOReBoKbjeR1D0fUvO4mbzaae/view?usp=sharing).

Install pretty midi before you start: 

```pip install pretty_midi```

To preprocess the midi files into tensor (input to the model), run

```python preprocess_large_midi_dataset.py -f /path/to/folder -n mydataset```
