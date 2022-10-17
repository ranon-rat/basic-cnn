package data

import (
	"image/png"
	"os"
)

type Data struct {
	Input    [][][]float32
	Expected []float32
}

func OpenImageAndGiveInfo(file string) (output [][][]float32) {
	f, _ := os.Open(file)
	img, _ := png.Decode(f)
	output = make([][][]float32, 1)
	output[0] = make([][]float32, img.Bounds().Max.Y)

	for y := 0; y < img.Bounds().Max.Y; y++ {
		output[0][y] = make([]float32, img.Bounds().Max.X)
		for x := 0; x < img.Bounds().Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			output[0][y][x] = float32(r+g+b) / (257 * 3)
		}
	}
	return
}

func OpenImages(folder string) (data []Data) {

	folders, _ := os.ReadDir(folder)
	for i, fol := range folders {
		if fol.IsDir() {
			ex := make([]float32, len(folders))
			ex[i] = 1
			files, _ := os.ReadDir(folder + "/" + fol.Name())
			for _, file := range files {
				data = append(data, Data{Input: OpenImageAndGiveInfo(folder + "/" + fol.Name() + "/" + file.Name()), Expected: ex})
			}

		}
	}
	return
}
