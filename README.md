# Activity-Detector
Toolbox that allow the user identify unsuppervised data on an audio signal.
Currently on development, only few tools are available.

##Intallation
### Requirements (manually install)
- python3
- numpy
- scipy
- librosa (audio reading and extracting features)
- matplotlib
- scikit-learn (use machine learning algorithms)

### With anaconda (recommanded)
Will coming in the future

## Usage

### Plot signal waveform and spectogram of a file

Will change in the future to be useful
```bash
  python src/plotting_clusters/plot_informations.py <audio filename>
```

### Compute mfcc features
Compute features from an audio file. The output is a csv file.
```bash
  python src/compute_features/mfcc.py <audio filename> <filename output> <window_features> <hop_time> <freq_min> <freq_max>
```
Times are in seconds and frequencies in Hz.

### Get clusters
Will come in the future
