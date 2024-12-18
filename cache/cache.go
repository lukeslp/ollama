package cache

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
)

type Options struct {
	Sequences []int
}

type Cache interface {
	Close()

	StartForward(ctx ml.Context, seqs []int) error

	Sub(i int) Cache
	Put(ctx ml.Context, key, value ml.Tensor, opts Options) (ml.Tensor, ml.Tensor, ml.Tensor)
	Remove(seq int, beginIndex, endIndex int)
}

type Simple struct {
	DType    ml.DType
	Capacity int

	curLayer     int
	sequences    []int
	pos          int
	curBatchSize int
	mask         ml.Tensor

	cacheCtx     ml.Context
	keys, values []ml.Tensor
}

func NewSimpleCache(backend ml.Backend, capacity int, dtype ml.DType) Cache {
	return &Simple{
		Capacity: capacity,
		DType:    dtype,
		// TODO(jessegross): This context is not sized appropriately
		cacheCtx: backend.NewContext(),
	}
}

func (c *Simple) Close() {
	c.cacheCtx.Close()
}

func (c *Simple) StartForward(ctx ml.Context, seqs []int) error {
	c.curBatchSize = len(seqs)
	c.pos = len(c.sequences)
	c.sequences = append(c.sequences, seqs...)

	if c.pos+c.curBatchSize >= c.Capacity {
		panic(fmt.Errorf("context length exceeded (length: %v)", c.Capacity))
	}

	var err error
	c.mask, err = c.buildMask(ctx, seqs)

	return err
}

func (c *Simple) buildMask(ctx ml.Context, seqs []int) (ml.Tensor, error) {
	// TODO(jessegross): This makes a number of simplifications, including assuming
	// causal attention, no padding, etc.

	curSize := c.pos + c.curBatchSize
	mask := make([]float32, c.curBatchSize*curSize)

	for i := range c.curBatchSize {
		for j := range curSize {
			if j > c.pos+i || seqs[i] != c.sequences[j] {
				mask[i*curSize+j] = float32(math.Inf(-1))
			}
		}
	}

	return ctx.FromFloatSlice(mask, curSize, c.curBatchSize)
}

func (c *Simple) Sub(i int) Cache {
	if i >= len(c.keys) {
		c.keys = append(c.keys, make([]ml.Tensor, i-len(c.keys)+1)...)
		c.values = append(c.values, make([]ml.Tensor, i-len(c.values)+1)...)
	}

	c.curLayer = i

	return c
}

func (c *Simple) Put(ctx ml.Context, key, value ml.Tensor, opts Options) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if c.curBatchSize != int(key.Dim(2)) {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, int(key.Dim(2))))
	}

	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		c.keys[c.curLayer] = c.cacheCtx.Zeros(c.DType, int(key.Dim(0)*key.Dim(1))*c.Capacity)
		c.values[c.curLayer] = c.cacheCtx.Zeros(c.DType, int(value.Dim(0)*value.Dim(1))*c.Capacity)
	}

	ctx.Forward(key.Copy(ctx, c.keys[c.curLayer].View(ctx, int(key.Stride(2))*c.pos, int(key.Dim(0)*key.Dim(1)*key.Dim(2)))))
	ctx.Forward(value.Copy(ctx, c.values[c.curLayer].View(ctx, int(value.Stride(2))*c.pos, int(value.Dim(0)*value.Dim(1)*value.Dim(2)))))

	n := int(key.Dim(2)) + c.pos

	key = c.keys[c.curLayer].View(ctx, 0,
		int(key.Dim(0)), int(key.Stride(1)),
		int(key.Dim(1)), int(key.Stride(2)),
		n,
	)

	value = c.values[c.curLayer].View(ctx, 0,
		int(value.Dim(0)), int(value.Stride(1)),
		int(value.Dim(1)), int(value.Stride(2)),
		n,
	)

	// TODO shift context if necessary

	return key, value, c.mask
}

func (c *Simple) Remove(seq int, beginIndex, endIndex int) {
	// TODO(jessegross): Some models don't support partial erasure
	for i := beginIndex; i < min(endIndex, len(c.sequences)); i++ {
		if c.sequences[i] == seq {
			c.sequences[i] = -1
		}
	}
}
