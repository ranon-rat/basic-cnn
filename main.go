package main

import (
	"github.com/ranon-rat/basic-cnn/brain"
	"github.com/ranon-rat/basic-cnn/data"
)

func main() {
	dat := data.OpenImages("shapes")
	net := brain.NewNetwork(200, 200, 7, 4, 5, []int{2, 2, 2, 2, 2}, []int{2})

	net.Train(dat, 2.5, 100)
	net.Save("model.json")
}
