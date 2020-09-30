package nnTest

import (
	"github.com/tadeuszjt/neuralnetwork"
	"testing"
)

func TestMakeNeuralNetwork(t *testing.T) {
	cases := []struct {
		numInputs, numOutputs, numLayers, numNeuronsPerLayer int
		numNeurons, numWeights                               int
	}{
		{
			1, 1, 1, 1,
			1, 4,
		},
		{
			8, 2, 3, 20,
			3 * 20, (8+1)*20 + (20+1)*20*2 + (20+1)*2,
		},
		{
			100, 10, 20, 200,
			20 * 200, (100+1)*200 + (200+1)*200*19 + (200+1)*10,
		},
	}

	for _, c := range cases {
		nn := nn.MakeNeuralNetwork(
			c.numInputs,
			c.numOutputs,
			c.numLayers,
			c.numNeuronsPerLayer,
		)

		if nn.NumInputs != c.numInputs {
			t.Errorf("expected: %v, got: %v", c.numInputs, nn.NumInputs)
		}

		if nn.NumOutputs != c.numOutputs {
			t.Errorf("expected: %v, got: %v", c.numOutputs, nn.NumOutputs)
		}

		if nn.NumLayers != c.numLayers {
			t.Errorf("expected: %v, got: %v", c.numLayers, nn.NumLayers)
		}

		if nn.NumNeuronsPerLayer != c.numNeuronsPerLayer {
			t.Errorf("expected: %v, got: %v", c.numNeuronsPerLayer, nn.NumNeuronsPerLayer)
		}

		if nn.NumNeuronsPerLayer != c.numNeuronsPerLayer {
			t.Errorf("expected: %v, got: %v", c.numNeuronsPerLayer, nn.NumNeuronsPerLayer)
		}

		expected := c.numInputs + c.numNeurons + c.numOutputs + c.numWeights
		if len(nn.Values) != expected {
			t.Errorf("expected: %v, got: %v", expected, len(nn.Values))
		}

		if len(nn.Inputs()) != c.numInputs {
			t.Errorf("expected: %v, got: %v", c.numInputs, len(nn.Inputs()))
		}

		if &nn.Inputs()[0] != &nn.Values[0] {
			t.Errorf("expected: %v, got: %v", &nn.Values[0], &nn.Inputs()[0])
		}

		if len(nn.Neurons()) != c.numNeurons {
			t.Errorf("expected: %v, got: %v", c.numNeurons, len(nn.Neurons()))
		}

		if &nn.Neurons()[0] != &nn.Values[c.numInputs] {
			t.Errorf("expected: %v, got: %v", &nn.Values[c.numInputs], &nn.Neurons()[0])
		}

		if len(nn.Outputs()) != c.numOutputs {
			t.Errorf("expected: %v, got: %v", c.numOutputs, len(nn.Outputs()))
		}

		if expected := &nn.Values[c.numInputs+c.numNeurons]; &nn.Outputs()[0] != expected {
			t.Errorf("expected: %v, got: %v", expected, &nn.Outputs()[0])
		}

		if expected, actual := c.numWeights, len(nn.Weights()); expected != actual {
			t.Errorf("expected: %v, got: %v", expected, actual)
		}

		if expected, actual := &nn.Values[c.numInputs+c.numNeurons+c.numOutputs], &nn.Weights()[0]; expected != actual {
			t.Errorf("expected: %v, got: %v", expected, actual)
		}
	}
}
