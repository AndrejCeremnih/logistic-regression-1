package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot/plotter"
)

const (
	epochs                           = 2000
	printEveryNthEpochs              = 100
	lrW                              = 0.5e-3
	lrB                              = 0.7
	inputPointsMinX, inputPointsMaxX = 0, 100
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func inference(inputs [][]float64, w []float64, b float64) []float64 {
	var res []float64
	for _, x := range inputs {
		res = append(res, sigmoid(dot(x, w)+b))
	}
	return res
}

func dot(a []float64, b []float64) float64 {
	var res float64
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, 2)
	m := float64(len(inputs))
	for i := range inputs {
		diff := p[i] - y[i]
		for j, x := range inputs[i] {
			dw[j] += x * diff / m
		}
		db += diff / m
	}
	return
}

func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	trainIndices := make(map[int]bool)
	rnd := rand.New(rand.NewSource(10))
	for i := 0; i < len(inputs)/5*4; i++ {
		idx := rnd.Intn(len(inputs))
		for trainIndices[idx] {
			idx = rnd.Intn(len(inputs))
		}
		trainIndices[idx] = true
	}
	for i := 0; i < len(inputs); i++ {
		if trainIndices[i] {
			xTrain = append(xTrain, inputs[i])
			yTrain = append(yTrain, y[i])
		} else {
			xTest = append(xTest, inputs[i])
			yTest = append(yTest, y[i])
		}
	}
	return
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	var res float64
	for i, x := range inputs {
		if y[i] == math.Round(sigmoid(dot(x, w)+b)) {
			res++
		}
	}
	return res / float64(len(y)) * 100
}

func main() {
	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("Logistic Regression")

	inputs, labels := readFromCSV("data/exams1.csv")
	xTrain, xTest, yTrain, yTest := split(inputs, labels)
	xys := make([]plotter.XYs, 2)
	for i := range inputs {
		if labels[i] == 0 {
			xys[0] = append(xys[0], plotter.XY{X: inputs[i][0], Y: inputs[i][1]})
		} else {
			xys[1] = append(xys[1], plotter.XY{X: inputs[i][0], Y: inputs[i][1]})
		}
	}
	var inputsScatter []*plotter.Scatter
	for i := range xys {
		tmp, _ := plotter.NewScatter(xys[i])
		inputsScatter = append(inputsScatter, tmp)
	}
	inputsScatter[0].Color = color.RGBA{255, 0, 0, 255}
	inputsScatter[1].Color = color.RGBA{0, 255, 0, 255}
	img := make(chan *image.RGBA, 1)
	render := func(x *image.RGBA) {
		select {
		case <-img:
			img <- x
		case img <- x:
		}
	}

	go func() {
		w := make([]float64, 2)
		var b float64
		for i := 0; i <= epochs; i++ {
			y := inference(xTrain, w, b)
			dw, db := dCost(xTrain, yTrain, y)
			for i := range dw {
				w[i] -= dw[i] * lrW
			}
			b -= db * lrB
			if i%printEveryNthEpochs == 0 {
				xs := []float64{inputPointsMinX, inputPointsMaxX}
				resLine, _ := plotter.NewLine(plotter.XYs{{X: xs[0], Y: -(w[0]*xs[0] + b) / w[1]}, {X: xs[1], Y: -(w[0]*xs[1] + b) / w[1]}})
				render(Plot(inputsScatter[0], inputsScatter[1], resLine))
				fmt.Printf(`Epoch #%d
dw: %.4f, db: %.4f
w: %.4f, b: %.4f
accuracy: %.2f
`, i, dw, db, w, b, accuracy(xTrain, yTrain, w, b))
			}
		}
		fmt.Printf("Accuracy: %.2f", accuracy(xTest, yTest, w, b))
	}()

	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}

func readFromCSV(filename string) ([][]float64, []float64) {
	inputs := make([][]float64, 100)
	var labels []float64
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	for i, row := range data {
		for j, col := range row {
			if j == 0 {
				v, err := strconv.ParseFloat(col, 64)
				if err != nil {
					log.Fatal(err)
				}
				inputs[i] = append(inputs[i], v)
			} else if j == 1 {
				v, err := strconv.ParseFloat(col, 64)
				if err != nil {
					log.Fatal(err)
				}
				inputs[i] = append(inputs[i], v)
			} else if j == 2 {
				v, err := strconv.ParseFloat(col, 64)
				if err != nil {
					log.Fatal(err)
				}
				labels = append(labels, v)
			}
		}
	}
	return inputs, labels
}
