# Gaze Tracking

This repository is a fork of [antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking)

## Installation

Clone this project:

```shell
git clone https://github.com/antoinelame/GazeTracking.git
```

### For Pip install

Install these dependencies (NumPy, OpenCV, Dlib):

```shell
pip install -r requirements.txt
```

> The Dlib library has four primary prerequisites: Boost, Boost.Python, CMake and X11/XQuartx. If you doesn't have them, you can [read this article](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) to know how to easily install them.

### For Anaconda install

Install these dependencies (NumPy, OpenCV, Dlib):

```shell
conda env create --file environment.yml
#After creating environment, activate it
conda activate GazeTracking
```

### Verify Installation

Run the demo:

```shell
python example.py
```
