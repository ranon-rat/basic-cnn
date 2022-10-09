package brain

import (
	"math/rand"
)

type Conv struct {
	Kernels        [][][]float32 `json:"kernels"`
	Bias           [][][]float32 `json:"bias"`
	ActivationFunc string        `json:"activationFunc"`
	Dimension      int           `json:"dimensions"`
}

func NewConv(filterNum int, widthAnHeightFilter int, activavtionFunc string) (layer Conv) {
	layer.Dimension = widthAnHeightFilter
	layer.ActivationFunc = activavtionFunc

	for i := 0; i < filterNum; i++ {
		layer.Kernels = append(layer.Kernels, make([][]float32, widthAnHeightFilter))

		for y := 0; y < widthAnHeightFilter; y++ {
			for x := 0; x < widthAnHeightFilter; x++ {
				layer.Kernels[i][y] = append(layer.Kernels[i][y], rand.Float32()-0.5)

			}
		}

	}
	return
}

func (c Conv) GetFullRegions(input [][][]float32) [][][][][]float32 {
	h, w := len(input[0]), len(input[0][0]) // the dimensions should have the same length lol, why you would put a different dimension input?

	// dimension , y coord, x coord, 2d region
	regions := make([][][][][]float32, len(input))

	for dim := 0; dim < len(input); dim++ {

		regions[dim] = make([][][][]float32, h)

		for y := 0; y < h; y++ {

			regions[dim][y] = make([][][]float32, w)

			for x := 0; x < w; x++ {

				row := make([][]float32, c.Dimension)
				// a region will contain the same size of  the kernel
				// we have a region for each pixel
				// so this will be much easy to understand
				for r := 0; r < c.Dimension; r++ {

					row[r] = make([]float32, c.Dimension)

					if (y+r)-((c.Dimension-1)/2) >= h || (y+r)-((c.Dimension-1)/2) < 0 {
						continue
					}
					for q := 0; q < c.Dimension; q++ {
						if x+q-((c.Dimension-1)/2) < 0 || x+q-((c.Dimension-1)/2) >= w {
							continue
						}
						row[r][q] = input[dim][(y+r)-((c.Dimension-1)/2)][(x+q)-((c.Dimension-1)/2)]
					}
				}
				regions[dim][y][x] = row
			}
		}
	}
	return regions

}

// The output will generate
func (c Conv) Foward(input [][][]float32) (output [][][]float32) {
	output = make([][][]float32, len(c.Kernels))
	regions := c.GetFullRegions(input)
	h, w := len(input[0]), len(input[0][0])
	for i := 0; i < len(c.Kernels); i++ {
		output[i] = make([][]float32, h)

		for dim := 0; dim < len(input); dim++ {

			for y := 0; y < h; y++ {

				output[i][y] = make([]float32, w)
				for x := 0; x < w; x++ {

					var sum float32

					for r := 0; r < c.Dimension; r++ {
						for q := 0; q < c.Dimension; q++ {
							// okay this is lazy

							// the thing that i made here is really simple
							// i just multiple the region of the dimension
							// for the kernel of the depth
							// with that i generate an output
							// its like the feedfoward algorithm but much easy to understand
							// and hard to implentate lol
							sum += regions[dim][y][x][r][q] * c.Kernels[i][r][q]

						}

					}

					output[i][y][x] += (sum / (float32(c.Dimension * c.Dimension)))
				}
			}

		}
	}
	for i := 0; i < len(output); i++ {
		for y := 0; y < h; y++ {

			for x := 0; x < w; x++ {

				output[i][y][x] += MathFuncs[c.ActivationFunc].Activate(output[i][y][x])
			}
		}

	}
	return
}

// I will take my time for thinking how i can make this
func (c Conv) BackProp(err [][][]float32, learningRate float32, output, input [][][]float32) ([][][]float32, [][][]float32) {

	// okay errors is the next layer
	// the gradient is for
	gradient := [][][]float32{}

	for d := 0; d < len(err); d++ {
		gradient = append(gradient, [][]float32{})
		for y := 0; y < len(err[0]); y++ {
			gradient[d] = append(gradient[d], []float32{})
			for x := 0; x < len(err[d][y]); x++ {
				// this should be enought for getting the gradient
				// i dont really know if its going to work but i hope it does
				gradient[d][y] = append(gradient[d][y], err[d][y][x]*MathFuncs[c.ActivationFunc].Derivate(output[d][y][x]))
			}
		}
	}
	regions := c.GetFullRegions(input)
	gradKernels := make([][][]float32, len(c.Kernels))

	for i := 0; i < len(c.Kernels); i++ {

		for dim := 0; dim < len(input); dim++ {
			gradKernels[i] = make([][]float32, len(regions[dim]))

			for y := 0; y < len(regions[dim]); y++ {
				gradKernels[i][y] = make([]float32, len(regions[dim][y]))

				for x := 0; x < len(regions[dim][y]); x++ {

					for r := 0; r < c.Dimension; r++ {
						for q := 0; q < c.Dimension; q++ {
							gradKernels[i][r][q] += gradient[dim][y][x] * regions[dim][y][x][r][q]
						}
					}

				}
			}
		}
	}
	errcp := input
	h, w := len(input[0]), len(input[0][0])
	regions = c.GetFullRegions(err)
	//fullConv(rotate 180 filter,lossGradient)
	for i := 0; i < len(c.Kernels); i++ {

		for dim := 0; dim < len(input); dim++ {

			for y := 0; y < h-(c.Dimension-1); y++ {

				errcp[i][y] = make([]float32, w-(c.Dimension-1))
				for x := 0; x < w-(c.Dimension-1); x++ {

					var sum float32

					for r := 0; r < c.Dimension; r++ {
						for q := 0; q < c.Dimension; q++ {

							sum += regions[dim][y][x][r][q] * c.Kernels[i][q][r]

						}

					}

					errcp[i][y][x] = sum / (float32(c.Dimension * c.Dimension))
				}
			}
		}
	}
	err = errcp
	return err, gradKernels

}
func (c *Conv) UpdateKernel(learningRate float32, kernelGradient [][][]float32) {
	for d := range c.Kernels {
		for y := 0; y < c.Dimension; y++ {
			for x := 0; x < c.Dimension; x++ {
				c.Kernels[d][y][x] -= kernelGradient[d][y][x] * learningRate
			}
		}
	}
}
