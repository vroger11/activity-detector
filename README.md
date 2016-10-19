# Activity-Detector
Toolbox that allow the user identify unsupervised data on an audio signal.
Currently on development, only few tools are available.

## Installation

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

### Plot activities (clusters) of data
#### Learn a VDPMM. Usage:
 ```bash
 python activity-detector/learn.py [-h] [-ml MAX_LEARN] [-v] [-l LOGFILE]
                folder_audio_in folder_out min_frequency max_frequency
 ```
Type ```python activity-detector/learn.py -h ``` for details.

#### Forward the model previously learned and save the results (Plot + file of clusters)
```bash
python activity-detector/forward.py [-h] [-v] [-l LOGFILE]
                folder_audio_in folder_model folder_out max_frequency
```

Type ```python activity-detector/forward.py -h ``` for details.

### Plot signal waveform and spectrogram of a file

```bash
  python activity-detector/plotting_clusters/plot_informations.py <audio filename>
```

### Compute mfcc features
Compute features from a folder of audio files. The output is csv files.
```bash
  python activity-detector/compute_features/mfcc.py [-v] [-l LOGFILE]
               folder_audio_in folder_output window_features hop_time
               [frequency_min] [frequency_max]
```

Type ```python activity-detector/compute_features/mfcc.py -h ``` for details.
