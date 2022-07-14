# Build a deep neural network for the keyword spotting (KWS) task with nnAudio GPU audio processing

### Step-by-step tutorial of how to use nnAudio to tackle the the keyword spotting (KWS) task


[nnAudio](https://ieeexplore.ieee.org/document/9174990) is a PyTorch tool for audio processing using GPU. Processing audio on the GPU **shortens the computation time by up to 100x.**

This tutorial will build a classifier on the Google speech commands dataset v2 for the Key Word Spotting (KWS) task.

There are a total of 12 different output classes in this KWS task. We chose to work with 10 out of all available 35 words, the remaining 25 words are grouped in the class 'unknown'. A class 'silence' is created from background noise.

We use [AudioLoader](https://github.com/KinWaiCheuk/AudioLoader) to access the speechcommands 12 classes dataset.

Our tutorial is available via Towards Data Science, feel free to check it out [here](https://towardsdatascience.com/build-a-deep-neural-network-for-the-keyword-spotting-kws-task-with-nnaudio-gpu-audio-processing-95b50018aaa8)

The tutorial below consists of four parts:
* [Part 1: Loading the dataset & simple linear model](#Part-1:-Loading-the-dataset-&-simple-linear-model)
* [Part 2: Training a Linear model with Trainable Basis Functions](#Part-2:-Training-a-Linear-model-with-Trainable-Basis-Functions)
* [Part 3: Evaluation of the resulting model](#Part-3:-Evaluation-of-the-resulting-model)
* [Part 4: Using more complex non-linear models](#Part-4:-Using-more-complex-non-linear-models)

### Part 1: Loading the dataset & simple linear model
In this tutorial, we will work with spectrograms while comparing the processing time between nnAudio GPU and librosa.

The result shows librosa took 27 mins for one epoch, however **nnAudio GPU only took around 30s to finish one epoch** which is 54x faster than librosa!

### Part 2: Training a Linear model with Trainable Basis Functions
We will demonstrate how to utilise nnAudio's trainable basis functions to build a powerful classifier.

nnAudio can calculate different types of spectrograms such as short-time Fourier transform (STFT), Mel-spectrogram, and onstant-Q transform (CQT) by leveraging PyTorch and GPU processing. In this project, we opted to work with Mel-spectrograms


```python
from nnAudio.features.mel import MelSpectrogram
MelSpectrogram(sr=16000, 
               n_fft=480,
               win_length=None,
               n_mels=n_mels, 
               hop_length=160,
               window='hann',
               center=True,
               pad_mode='reflect',
               power=2.0,
               htk=False,
               fmin=0.0,
               fmax=None,
               norm=1,
               trainable_mel=True,
               trainable_STFT=True,
               verbose=True)
```

### Part 3: Evaluation of the resulting model
We will evaluate the model performance and do visualization by using the trained model weight from tutorial 2.

### Part 4: Using more complex non-linear models
After following part 1–3 of the tutorial, you now have a big picture overview of how to use nnAudio with trainable basis functions.
In this tutorial, Broadcasting-residual network (BC_ResNet) will be used for demonstration on how nnAudio is applied in more complex model.


# References
1. K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, ["nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," in IEEE Access](https://ieeexplore.ieee.org/document/9174990) , vol. 8, pp. 161981–162003, 2020, doi: 10.1109/ACCESS.2020.3019084.

1. For nnAudio source code, you can refer to https://github.com/KinWaiCheuk/nnAudio

1. For nnAudio documentation, you can refer to https://kinwaicheuk.github.io/nnAudio/index.html

1. Heung, K. Y., Cheuk, K. W., & Herremans, D. (2022). [Understanding Audio Features via Trainable Basis Functions](https://arxiv.org/pdf/2204.11437.pdf). arXiv preprint arXiv:2204.11437. (Research article on trainable basis functions)



