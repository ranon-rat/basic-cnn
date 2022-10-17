package brain

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"

	"github.com/ranon-rat/basic-cnn/data"
)

type Network struct {
	ConvLayers []Conv    `json:"conv"`
	SoftLayers []SoftMax `json:"soft"`
	Pooling    []int     `json:"pooling"`
}

func Flatten(input [][][]float32) (output []float32) {
	for d := 0; d < len(input); d++ {
		for y := 0; y < len(input[d]); y++ {
			for x := 0; x < len(input[d][y]); x++ {

				output = append(output, input[d][y][x])
			}
		}
	}
	return
}
func Unflatten(input []float32, width, height int) (output [][][]float32) {
	output = append(output, make([][]float32, height))
	output[0][0] = make([]float32, width)
	x, y, z := 0, 0, 0
	for i := 0; i < len(input); i++ {

		if y+1 == height && x == width {
			z++
			output = append(output, make([][]float32, height))
			output[z][0] = make([]float32, width)
			x, y = 0, 0
		}
		output[z][y][x] = input[i]

		x++
		if x == width && y+1 != height {
			y++
			output[z][y] = make([]float32, width)
			x = 0
		}
	}

	return
}

/* the layersSize is for the dense network you dont put the input size,just the output
 */
func NewNetwork(inputHeight, inputWidth, filterNum, widthAnHeightFilter, layersConv int, scaling []int, layerSize []int) (net Network) {
	net.Pooling = scaling
	w := inputWidth
	h := inputHeight
	for i := 0; i < layersConv; i++ {
		net.ConvLayers = append(net.ConvLayers, NewConv(filterNum, widthAnHeightFilter, "relu"))
		h /= scaling[i]
		w /= scaling[i]
	}
	layes := []int{h * w * filterNum}
	layes = append(layes, layerSize...)
	for i := 0; i < len(layerSize); i++ {
		net.SoftLayers = append(net.SoftLayers, NewSoftMaxLayer(layes[i], layes[i+1]))
	}
	return
}
func (net Network) Foward(input [][][]float32) (layersC [][][][]float32, denseLayers [][]float32, height, width []int) {

	layersC = make([][][][]float32, len(net.ConvLayers)+1)

	layersC[0] = input

	for l := 0; l < len(net.ConvLayers); l++ {

		lays := net.ConvLayers[l].Foward(layersC[l])
		// I need this for reescaling it later for the training part
		height = append(height, len(lays[0]))
		width = append(width, len(lays[0][0]))
		// this is going to rescale it so its going to be more stable I think maybe Idk a
		layersC[l+1] = Pooling(lays, net.Pooling[l])

	}
	// this is used for passing the output of the convolutional layers to the dense network
	height = append(height, len(layersC[len(layersC)-1][0]))
	width = append(width, len(layersC[len(layersC)-1][0][0]))
	// this is really basic
	denseLayers = make([][]float32, len(net.SoftLayers)+1)

	denseLayers[0] = Flatten(layersC[len(net.ConvLayers)])
	for l := 0; l < len(net.SoftLayers); l++ {

		// and jusst that
		denseLayers[l+1] = net.SoftLayers[l].Foward(denseLayers[l])
	}
	return

}

func (net Network) BackProp(Expected []float32, denseLayers [][]float32, convLayers [][][][]float32, width, height []int) (wds [][][]float32, bds [][]float32, kds [][][][]float32) {
	// so , you supose to use all the stuff that the forward function return so this is going to be really basic
	// the error is used for the gradient descent

	err := make([]float32, len(Expected))
	//this is going to be used for the dense network for updating the layers and all that stuff
	wds = make([][][]float32, len(denseLayers)-1)
	bds = make([][]float32, len(denseLayers)-1)

	// loss function
	for i := 0; i < len(Expected); i++ {
		err[i] = (denseLayers[len(net.SoftLayers)][i]) - Expected[i]
	}
	// this is really basic I just do this for getting everything that I need
	for l := len(net.SoftLayers) - 1; l >= 0; l-- {
		wds[l], bds[l], err = net.SoftLayers[l].Backprop(err, denseLayers[l], denseLayers[l+1])

	}
	// how i Said I need to use this for rescale it
	// in this case i unflat the error array an I get this

	errConv := Unflatten(err, width[len(net.ConvLayers)], height[len(net.ConvLayers)])
	kds = make([][][][]float32, len(net.ConvLayers))
	for l := len(net.ConvLayers) - 1; l >= 0; l-- {

		dep := Depooling(convLayers[l+1], height[l], width[l], net.Pooling[l])
		errConv = Depooling(errConv, height[l], width[l], net.Pooling[l])

		errConv, kds[l] = net.ConvLayers[l].BackProp(errConv, dep, convLayers[l])
	}
	return
}

func (net Network) Train(data []data.Data, learningRate float32, epochs int) {
	for e := 0; e < epochs; e++ {
		index := rand.Intn(len(data))
		cl, dl, h, w := net.Foward(data[index].Input)
		wds, bds, kds := net.BackProp(data[index].Expected, dl, cl, w, h)
		if e%10 == 0 {
			fmt.Printf("cost: %f9.5f |epoch: %d ", cost(data[index].Expected, dl[len(net.SoftLayers)]), e)
		}
		for l := 0; l < len(net.SoftLayers); l++ {
			net.SoftLayers[l].Update(wds[l], bds[l], learningRate)

		}
		for l := 0; l < len(net.ConvLayers); l++ {
			net.ConvLayers[l].UpdateKernel(kds[l], learningRate)
		}
	}

}
func (net Network) Save(file string) {
	f, _ := os.CreateTemp("./", file)
	json.NewEncoder(f).Encode(net)
}

func OpenModel(file string) (net Network) {
	f, _ := os.Open(file)
	json.NewDecoder(f).Decode(&net)
	return
}
