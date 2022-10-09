package brain

type Network struct {
	ConvLayers []Conv
	SoftLayers []SoftMax
	Pooling    []int
}

func NewNetwork(inputHeight, inputWidth, filterNum, widthAnHeightFilter, layersConv, output int, activationFunc []string, scaling []int) (net Network) {
	net.Pooling = scaling
	input := inputHeight * inputWidth
	for i := 0; i < layersConv; i++ {
		net.ConvLayers = append(net.ConvLayers, NewConv(filterNum, widthAnHeightFilter, "relu"))
		input = (input * filterNum) / scaling[i]
	}
	for i := 0; i < len(activationFunc); i++ {
		net.SoftLayers = append(net.SoftLayers, NewSoftMaxLayer(input, output))
	}
	return
}
