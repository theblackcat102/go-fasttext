package fasttext

// #cgo CXXFLAGS: -I${SRCDIR}/fastText/src -I${SRCDIR} -std=c++14
// #cgo LDFLAGS: -lstdc++
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits.h"
import "C"

import (
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

// CosineSimilarity calculates the cosine similarity between two Vectors.
func (handle *Model) CosineSimilarity(a string, b string) float32 {
	cA := C.CString(a)
	defer C.free(unsafe.Pointer(cA))
	cB := C.CString(b)
	defer C.free(unsafe.Pointer(cB))
	sum := C.VecSimilarity(handle.handle, cA, cB)
	return float32(sum)
}