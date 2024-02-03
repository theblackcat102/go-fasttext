// +build ignore

package main

import (
	fasttext "github.com/theblackcat102/go-fasttext"
)

func main() {
    modelPath := "test.bin"
    // Open the FastText model
    model := fasttext.Open(modelPath)
    A := "Alice"
    B := "Queen"
    ab := model.CosineSimilarity(A, B)
    pp.Println(ab)
    defer model.Close()
}
