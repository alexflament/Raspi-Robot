# Multi-Layer Perceptron Library

MLP is a multi-layer perceptron library. It is designed in C++, and will facilitate the creation and optimization of multi-layer neural networks ( perceptrons).

The following example load data from the file `sin.dat` and calibrate a three layers perception with : 1 neuron on the input layer, 5 on the hidden layer and 1 on the output layer.

```
  int layers[] = {1,5,1};
  MultiLayerPerceptron mlp(3,layers);
  mlp.Run("sin.dat",500);
```

