package main

import (
	"encoding/json"
	"os"

	"github.com/ranon-rat/basic-cnn/brain"
)

func main() {
	example := [][][]float32{
		{
			{255, 0, 0, 0},
			{0, 255, 0, 0},
			{0, 0, 255, 0},
			{0, 0, 0, 255},
		},
		{
			{255, 0, 0, 0},
			{0, 255, 0, 255},
			{0, 0, 255, 0},
			{0, 255, 0, 255},
		},
	}
	//c := brain.NewConv(3, 3, "relu")
	f, _ := os.OpenFile("a.json", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644) //
	json.NewEncoder(f).Encode(brain.Pooling(example, 2))

}
