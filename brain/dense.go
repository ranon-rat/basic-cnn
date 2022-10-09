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

func (net SoftMax) Foward(Input []float32) (output []float32) {
	output = make([]float32, len(net.Bias))
	copy(output, net.Bias)
	for n := 0; n < len(net.Weights); n++ {

		for c := 0; c < len(net.Weights[n]); c++ {
			output[c] += Input[n] * net.Weights[n][c]
		}
	}
	for n := 0; n < len(output); n++ {
		output[n] = MathFuncs[net.ActivationFunc].Activate(output[n])
	}
	return output
}

func (net SoftMax) Backprop(output []float32, expected []float32) ([][]float32, []float32, []float32) {
	bd := make([]float32, len(net.Bias))
	wd := make([][]float32, len(net.Weights))

	errors := make([]float32, len(expected))

	// I dont need to explain this one
	for i, n := range output {
		errors[i] = n - expected[i]
	}

	for i := range net.Bias {
		//gradient=errors*dy/dx(fx)(layer[l+1])
		bd[i] += errors[i] * MathFuncs[net.ActivationFunc].Derivate(output[i])
	}
	//layer_t *gradient
	for n := 0; n < len(wd); n++ {
		wd[n] = make([]float32, len(net.Weights[n]))

		for i := range net.Weights[n] {

			wd[n][i] += output[n] * (bd[i])
		}
	}

	errorcp := make([]float32, len(net.Weights))
	// errors=weights_t*errors
	for i := range net.Weights {

		var err float32 = 0.0
		for j := range errors {
			err += net.Weights[i][j] * errors[j]
		}
		errorcp[i] = err
	}
	errors = errorcp

	return wd, bd, errors

}

func (net *SoftMax) UpdateWeights(wd [][]float32, bd []float32, lr float32) {
	for n := 0; n < len(net.Weights); n++ {
		for c := 0; c < len(net.Bias); c++ {
			net.Weights[n][c] -= wd[n][c] * lr
		}
	}
	for n := 0; n < len(net.Bias); n++ {
		net.Bias[n] -= bd[n] * lr
	}
}
