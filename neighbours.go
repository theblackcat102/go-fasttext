package fasttext

import (
	"math"
	"sort"
)

type Neighbor struct {
	Index       int     `json:"index"`
  	Name       string  `json:"name"`
	Probability float32 `json:"probability"`
}

type Neighbors []Neighbor

// Len is the number of elements in the collection.
func (p Neighbors) Len() int {
	return len(p)
}

// Less reports whether the element with
// index i should sort before the element with index j.
func (p Neighbors) Less(i, j int) bool {
	pi := p[i].Probability
	pj := p[j].Probability
	return !(pi < pj || math.IsNaN(float64(pi)) && !math.IsNaN(float64(pj)))
}

// Swap swaps the elements with indexes i and j.
func (p Neighbors) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p Neighbors) Sort() {
	sort.Sort(p)
}
