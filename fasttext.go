package fasttext

// #cgo CXXFLAGS: -I${SRCDIR}/fastText/src -I${SRCDIR} -std=c++14
// #cgo LDFLAGS: -lstdc++
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits.h"
import "C"

import (
	"math"
	"encoding/json"
	"unsafe"
)

// A model object. Effectively a wrapper
// around the C fasttext handle
type Model struct {
	path   string
	handle C.FastTextHandle
}

// Opens a model from a path and returns a model
// object
func Open(path string) *Model {
	// fmt.Println("something")
	// create a C string from the Go string
	cpath := C.CString(path)
	// you have to delete the converted string
	// See https://github.com/golang/go/wiki/cgo
	defer C.free(unsafe.Pointer(cpath))

	return &Model{
		path:   path,
		handle: C.NewHandle(cpath),
	}
}

// Closes a model handle
func (handle *Model) Close() error {
	if handle == nil {
		return nil
	}
	C.DeleteHandle(handle.handle)
	return nil
}

// Performs model nearest neighbour search
func (handle *Model) Neighbor(query string, k int32) (Neighbors, error) {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))
	ck := C.int(k)
	r := C.Neighbor(handle.handle, cquery, ck)
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	neighbours := []Neighbor{}
	err := json.Unmarshal([]byte(js), &neighbours)
	if err != nil {
		return nil, err
	}

	return neighbours, nil
}

// Performs model prediction
func (handle *Model) Predict(query string) (Predictions, error) {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))

	// Call the Predict function defined in cbits.cpp
	// passing in the model handle and the query string
	r := C.Predict(handle.handle, cquery)
	// the C code returns a c string which we need to
	// convert to a go string
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	// unmarshal the json results into the predictions
	// object. See https://blog.golang.org/json-and-go
	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}

	return predictions, nil
}

func (handle *Model) Analogy(A string, B string, C string, k int32) (Analogs, error) {
	// A + B - C
	cA := C.CString(A)
	defer C.free(unsafe.Pointer(cA))

	cB := C.CString(B)
	defer C.free(unsafe.Pointer(cB))

	cC := C.CString(C)
	defer C.free(unsafe.Pointer(cC))

	ck := C.int(k)

	r := C.Analogy(handle.handle, cA, cB, cC, ck)
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	analogies := []Analog{}
	err := json.Unmarshal([]byte(js), &analogies)
	if err != nil {
		return nil, err
	}

	return analogies, nil
}

func (handle *Model) Wordvec(query string) (Vectors, error) {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))

	r := C.Wordvec(handle.handle, cquery)
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	vectors := []Vector{}
	err := json.Unmarshal([]byte(js), &vectors)
	if err != nil {
		return nil, err
	}

	return vectors, nil
}

// DotProduct calculates the dot product of two Vectors.
func DotProduct(v1, v2 Vectors) (float32, error) {
	if len(v1) != len(v2) {
		return 0, fmt.Errorf("vectors must be of the same length")
	}

	var product float32
	for i, v := range v1 {
		product += v.Element * v2[i].Element
	}
	return product, nil
}

func Norm(v Vectors) float32 {
	var sum float32
	for _, vector := range v {
		sum += vector.Element * vector.Element
	}
	return float32(math.Sqrt(float64(sum)))
}

// CosineSimilarity calculates the cosine similarity between two Vectors.
func (handle *Model) CosineSimilarity(a string, b string) (float32, error) {
	vA, err = Model.Wordvec(a)
	if err != nil {
		return 0, err
	}
	vB, err = Model.Wordvec(b)
	if err != nil {
		return 0, err
	}
	dotProduct, err := DotProduct(vA, vB)
	if err != nil {
		return 0, err
	}
	normV1 := Norm(v1)
	normV2 := Norm(v2)
	var sum float32 = 0
	for i := range a {
		sum += a[i].Element * b[i].Element
	}
	return sum, nil
}