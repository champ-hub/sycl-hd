## Compiling and Running the included examples

To compile and execute any example the process is identical. To compile:

```bash
make [EXAMPLE]
```

To run:
```bash
cd examples/[EXAMPLE]/
./[EXAMPLE] -vs [VECTORSIZE]
cd ../../
```

where `[VECTORSIZE]` indicates the hypervector size to use and `[EXAMPLE]` indicates the example to be compiled and executed. Currently available options are:

- [*voicehd*](voicehd/README.md) - VoiceHD speech recognition application;
- [*mnist*](mnist/Readme.md) - Image classification of the MNIST dataset;
- [*language*](language/Readme.md) - European language recognition application;
- [*hdna*](hdna/Readme.md) - HDNA genome sequencing application;
- [*emg*](emg/README.md) - Gesture recognition application;
- [*cmapss*](cmapss/README.md) - Jet Engine Remaining Useful Life Estimation application.
- [*dofmex*](dofmex/README.md) - Dollar of Mexico
- [*cifar10*](cifar10/README.md)

After the successful execution of an example application, the expected outcome for most is something similar to:

```
Results
Library Version: v0.3
Finished at: Wed Oct 26 02:14:56 2022
Accelerator: Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz
Success Rate: 87.5561%
Training Time: 6.2806s
Testing Time: 1.94374s
Vector Size: 10000
```

