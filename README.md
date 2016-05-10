# Activity-Detector
Toolbox that allow the user identify unsupervised data on an audio signal.
Currently on development, only few tools are available.

##Installation

### With anaconda (recommended)
First install anaconda, example use [miniconda](http://conda.pydata.org/miniconda.html) (ligth version of anaconda).

Next, create the environment with the .yml file of this project.
```bash
    conda env create -f conda_environment.yml
```

Don't forget to activate it using this project.

### Requirements (for manually installing)
- python3
- numpy
- scipy
- librosa (audio reading and extracting features)
- matplotlib
- scikit-learn (machine learning algorithms)

## Usage

### Plot signal waveform and spectrogram of a file

Will change in the future to be useful
```bash
  python src/plotting_clusters/plot_informations.py <audio filename>
```

### Compute mfcc features
Compute features from a folder of audio files. The output is csv files.
```bash
  python src/compute_features/mfcc.py folder_audio_in folder_output window_features hop_time
               [frequency_min] [frequency_max]
```

Type ```python src/compute_features/mfcc.py -h ``` for details.

### Get clusters
Will come in the future

### Plot activities (clusters) of data
Learn a VDPMM and plot the clusters founds from it. In the future it will evolve. Usage:
 ```bash
 python src/main.py folder_audio_in
 ```
Type ```python src/main.py -h ``` for details.