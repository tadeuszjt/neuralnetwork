package nn

import (
	"math"
	"math/rand"
)

const activationScalar = 4

type NeuralNetwork struct {
	// [i0 .. in]
	// [n0 .. nn]
	// [o0 .. on]
	// [w0 .. wn]
	Values             []float32
	NumInputs          int
	NumOutputs         int
	NumLayers          int
	NumNeuronsPerLayer int
}

func MakeNeuralNetwork(numInputs, numOutputs, numLayers, numNeuronsPerLayer int) NeuralNetwork {
	if numLayers < 1 {
		panic("must have at least one layer")
	}

	numWeights := (numInputs + 1) * numNeuronsPerLayer
	numWeights += (numNeuronsPerLayer + 1) * numNeuronsPerLayer * (numLayers - 1)
	numWeights += (numNeuronsPerLayer + 1) * numOutputs

	numVals := numInputs + (numLayers * numNeuronsPerLayer) + numOutputs + numWeights

	return NeuralNetwork{
		Values:             make([]float32, numVals),
		NumInputs:          numInputs,
		NumOutputs:         numOutputs,
		NumLayers:          numLayers,
		NumNeuronsPerLayer: numNeuronsPerLayer,
	}
}

func (nn *NeuralNetwork) Inputs() []float32 {
	return nn.Values[:nn.NumInputs]
}

func (nn *NeuralNetwork) Neurons() []float32 {
	idx := nn.NumInputs
	return nn.Values[idx : idx+nn.NumLayers*nn.NumNeuronsPerLayer]
}

func (nn *NeuralNetwork) Outputs() []float32 {
	idx := nn.NumInputs + nn.NumLayers*nn.NumNeuronsPerLayer
	return nn.Values[idx : idx+nn.NumOutputs]
}

func (nn *NeuralNetwork) Weights() []float32 {
	idx := nn.NumInputs + nn.NumLayers*nn.NumNeuronsPerLayer + nn.NumOutputs
	return nn.Values[idx:]
}

func (nn *NeuralNetwork) Process() {
	inputs := nn.Inputs()
	neurons := nn.Neurons()
	outputs := nn.Outputs()
	weights := nn.Weights()

	numNeurons := nn.NumLayers * nn.NumNeuronsPerLayer

	w := 0
	n := 0

	for ; n < nn.NumNeuronsPerLayer; n++ {
		activation := float32(0.0)

		for i := 0; i < len(inputs); i++ {
			activation += inputs[i] * weights[w]
			w++
		}

		activation += weights[w]
		w++
		neurons[n] = 1 / (1 + float32(math.Exp(-activationScalar*float64(activation))))
	}

	for ; n < numNeurons; n++ {
		p := ((n / nn.NumNeuronsPerLayer) - 1) * nn.NumNeuronsPerLayer
		activation := float32(0.0)

		for i := 0; i < nn.NumNeuronsPerLayer; i++ {
			activation += neurons[p+i] * weights[w]
			w++
		}

		activation += weights[w]
		w++

		neurons[n] = 1 / (1 + float32(math.Exp(-activationScalar*float64(activation))))
	}

	p := (nn.NumLayers - 1) * nn.NumNeuronsPerLayer
	for o := 0; o < nn.NumOutputs; o++ {
		activation := float32(0.0)

		for i := 0; i < nn.NumNeuronsPerLayer; i++ {
			activation += neurons[p+i] * weights[w]
			w++
		}

		activation += weights[w]
		w++

		outputs[o] = 1 / (1 + float32(math.Exp(-activationScalar*float64(activation))))
	}
}

func (nn *NeuralNetwork) RandomiseWeights() {
	for i := range nn.Weights() {
		norm := rand.NormFloat64() * 0.4
		if norm > 1.0 {
			norm = 1.0
		} else if norm < -1.0 {
			norm = -1.0
		}
		nn.Weights()[i] = float32(norm)
	}
}
