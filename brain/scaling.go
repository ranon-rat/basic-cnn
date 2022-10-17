package brain

func GetRegions(input [][][]float32, scale int) [][][][][]float32 {
	output := make([][][][][]float32, len(input))
	hOut, wOut := len(input[0])/scale, len(input[0][0])/scale
	for dim := 0; dim < len(output); dim++ {
		output[dim] = make([][][][]float32, hOut)
		for y := 0; y < hOut; y++ {
			output[dim][y] = make([][][]float32, wOut)

			for x := 0; x < wOut; x++ {
				row := [][]float32{}
				for c := 0; c < scale; c++ {
					row = append(row, input[dim][y*scale+c][x*scale:x*scale+scale])
				}
				output[dim][y][x] = row

			}
		}
	}
	return output
}

// average pooling lol
// only return the input scaled
func Pooling(input [][][]float32, scale int) [][][]float32 {
	output := make([][][]float32, len(input))
	regions := GetRegions(input, scale)
	hOut, wOut := len(input[0])/scale, len(input[0][0])/scale
	for dim := 0; dim < len(output); dim++ {
		output[dim] = make([][]float32, hOut)
		for y := 0; y < hOut; y++ {
			output[dim][y] = make([]float32, wOut)

			for x := 0; x < wOut; x++ {
				var sum float32
				for c := 0; c < scale; c++ {
					for k := 0; k < scale; k++ {
						sum += regions[dim][x][y][c][k]

					}
				}
				output[dim][y][x] = sum / (float32(scale * scale))

			}
		}
	}
	return output
}

// with this i return how the output could look like
// this is for the backpropagation
// really simple i think

func Depooling(output [][][]float32, oriH, oriW, scale int) [][][]float32 {

	input := make([][][]float32, len(output))
	for dim := 0; dim < len(output); dim++ {
		input[dim] = make([][]float32, oriH)
		for y := 0; y < oriH; y++ {

			input[dim][y] = make([]float32, oriW)
			for x := 0; x < oriW; x++ {

				input[dim][y][x] = output[dim][y/scale][x/scale]
			}
		}
	}
	return input
}
