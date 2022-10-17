package brain

import (
	"math/rand"
)

type SoftMax struct {
	Bias    []float32   `json:"bias"`
	Weights [][]float32 `json:"weights"`
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
	for n := 0; n < len(net.Weights); n++ {

		for c := 0; c < len(net.Weights[n]); c++ {
			output[c] += Input[n] * net.Weights[n][c]
		}
	}
	for n := 0; n < len(output); n++ {
		output[n] = softMax(output, output[n]+net.Bias[n])
	}
	return output
}

func (net SoftMax) Backprop(errors []float32, input []float32, output []float32) (wd [][]float32, bd []float32, err []float32) {
	bd = make([]float32, len(net.Bias))
	wd = make([][]float32, len(net.Weights))

	// I dont need to explain this one

	for i := range net.Bias {
		//gradient=errors*dy/dx(fx)(layer[l+1])
		bd[i] += errors[i] * devSoftMax(output, i)
	}
	//layer_t *gradient
	for n := 0; n < len(wd); n++ {
		wd[n] = make([]float32, len(net.Weights[n]))

		for i := range net.Weights[n] {

			wd[n][i] += input[n] * (bd[i])
		}
	}

	err = make([]float32, len(net.Weights))
	// errors=weights_t*errors
	for i := range input {

		var e float32 = 0.0
		for j := range errors {
			e += net.Weights[i][j] * errors[j]
		}
		err[i] = e
	}
	return wd, bd, err

}

func (net *SoftMax) Update(wd [][]float32, bd []float32, lr float32) {
	for n := 0; n < len(net.Weights); n++ {
		for c := 0; c < len(net.Bias); c++ {
			net.Weights[n][c] -= wd[n][c] * lr
		}
	}
	for n := 0; n < len(net.Bias); n++ {
		net.Bias[n] -= bd[n] * lr
	}
}
