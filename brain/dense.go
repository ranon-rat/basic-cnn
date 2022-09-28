package brain

import "math/rand"

type SoftMax struct {
	Bias           []float32
	Weights        [][]float32
	ActivationFunc string
}

func NewSoftMaxLayer(input, output int) (layer SoftMax) {
	layer.Weights = make([][]float32, input)
	layer.Bias = make([]float32, output)
	for n := 0; n < input; n++ {
		layer.Weights[n] = make([]float32, output)
		for c := 0; c < output; c++ {
			layer.Weights[n][c] = rand.Float32() - 0.5
		}
	}
	for n := 0; n < output; n++ {
		layer.Bias[n] = rand.Float32() - 0.5
	}
	return layer
}
