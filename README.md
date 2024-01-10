# Go-FastText

Golang bindings to the fasttext library. Added nearest neighbor query function

```
package main

import (
    "fmt"
    "github.com/k0kubun/pp"
    fasttext "github.com/theblackcat102/go-fasttext"
)

func main() {
    // Replace with the path to your .bin model file
    modelPath := "<model path>"
    model := fasttext.Open(modelPath)
    defer model.Close()

    word := "YOLO"
    mostSimilars, err := model.Neighbor(word, 10)
    if err != nil {
        fmt.Println(err)
        return
    }
    pp.Println(mostSimilars)
}
```

## Usage

To perform a prediction on a model, use the following command

```
go run main.go prediction -m [model_path] [query]
```

For example

```
go run main.go predict -m ~/Downloads/ag_news.bin chicken
```
