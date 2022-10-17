package brain

import "math"

type ActivationFunc struct {
	Derivate func(float32) float32
	Activate func(float32) float32
}

func relu(x float32) float32 {
	return float32(math.Max(0, float64(x)))

}
func devRelu(x float32) float32 {
	out := 0.0
	if x > 0 {
		out = 1
	}
	return float32(out)
}
func softMax(inputs []float32, input float32) (output float32) {
	sum := 0.0
	for i := 0; i < len(inputs); i++ {
		sum += math.Exp(float64(inputs[i]))
	}
	return input / float32(sum)

}
func devSoftMax(inputs []float32, inputIndex int) (output float32) {
	var sum float32 = 0.0
	for i := 0; i < len(inputs); i++ {
		if i == inputIndex {
			continue
		}
		sum += (inputs[i])

	}
	return inputs[inputIndex] * (1 - sum)
}
func cost(target []float32, output []float32) float32 {
	err := 0.0
	for i := range target {
		err += math.Pow(float64(output[i]-target[i]), 2)
	}
	return float32(err)
}

var (
	MathFuncs = map[string]ActivationFunc{

		"relu": {
			Derivate: devRelu,
			Activate: relu,
		},
	}
)
