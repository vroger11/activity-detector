# Activity-Detector
Toolbox for unsupervised audio signal learning.
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
#### Learn a model:
 ```bash
 python activity-detector/learn.py [-h] [-ml MAX_LEARN] [-v] [-l LOGFILE]
                folder_audio_in folder_out config
 ```
 ``` config ``` is a JSON file, it contains model, feature and other details. Look at config/example_config.json for an example.

Type ```python activity-detector/learn.py -h ``` for details.

#### Forward the model previously learned
```bash
python activity-detector/forward.py [-h] [-v] [-l LOGFILE]
                  folder_audio_in folder_model folder_out
```

Type ```python activity-detector/forward.py -h ``` for details.

#### Plot forwarded data (Plot of clusters)
```bash
python activity-detector/plot_forwarded.py [-h] [-v] [-l LOGFILE]
                         folder_forwarded folder_out max_frequency
```

Type ```python activity-detector/plot_forwarded.py -h``` for details.

### Plot signal waveform and spectrogram of a file

```bash
  python activity-detector/plotting_clusters/plot_informations.py <audio filename>
```
