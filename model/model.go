package model

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log/slog"
	"os"
	"reflect"
	"strconv"
	"strings"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend"
)

type Cache struct {
	cache.Cache
	cache.Options
}

func (c Cache) StartForward(ctx ml.Context, seqs []int) error {
	if c.Cache != nil {
		return c.Cache.StartForward(ctx, seqs)
	}

	return nil
}

func (c Cache) Sub(i int) Cache {
	if c.Cache != nil {
		return Cache{
			Cache:   c.Cache.Sub(i),
			Options: c.Options,
		}
	}

	return c
}

func (c Cache) Put(ctx ml.Context, key, value ml.Tensor, opts cache.Options) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if c.Cache != nil {
		return c.Cache.Put(ctx, key, value, opts)
	}

	// TODO(jessegross): Mask can't be nil but we should just remove the ability to run without a cache
	return key, value, nil
}

type Options struct {
	inputs    []int32
	positions []int32
	outputs   []int32

	sequences []int

	Images []image.Image

	Cache
}

func (opts Options) Inputs() []int32 {
	return opts.inputs
}

func (opts Options) Positions() []int32 {
	return opts.positions
}

func (opts Options) Outputs() []int32 {
	return opts.outputs
}

type OptionsFunc func(Model, *Options)

func WithInputIDs(ids []int32) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.inputs = ids
	}
}

func WithPositions(pos []int32) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.positions = pos
	}
}

func WithOutputs(outputs []int32) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.outputs = outputs
	}
}

func WithSequences(seqs []int) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.sequences = seqs
		opts.Cache.Sequences = seqs
	}
}

func WithImage(img image.Image) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.Images = append(opts.Images, img)
	}
}

func WithCache(c cache.Cache) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.Cache = Cache{
			Cache: c,
			Options: cache.Options{
				Sequences: opts.sequences,
			},
		}
	}
}

type Base struct {
	b ml.Backend
}

func (m *Base) Backend() ml.Backend {
	return m.b
}

func (m *Base) SetBackend(b ml.Backend) {
	m.b = b
}

type Model interface {
	Forward(ml.Context, Options) (ml.Tensor, error)

	Backend() ml.Backend
	SetBackend(ml.Backend)
}

var models = make(map[string]func(ml.Config) (Model, error))

func Register(name string, f func(ml.Config) (Model, error)) {
	if _, ok := models[name]; ok {
		panic("model: model already registered")
	}

	models[name] = f
}

func New(s string) (Model, error) {
	r, err := os.Open(s)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	b, err := ml.NewBackend(r)
	if err != nil {
		return nil, err
	}

	arch := b.Config().Architecture()
	f, ok := models[arch]
	if !ok {
		return nil, fmt.Errorf("unsupported model architecture %q", arch)
	}

	m, err := f(b.Config())
	if err != nil {
		return nil, err
	}

	if err := loadTensors(b, m); err != nil {
		return nil, err
	}

	m.SetBackend(b)
	return m, nil
}

var mlTensorType = reflect.TypeOf((*ml.Tensor)(nil)).Elem()

func loadTensors(b ml.Backend, m any, tensorPath ...string) error {
	t := reflect.TypeOf(m)
	v := reflect.ValueOf(m)

	if t.Kind() == reflect.Pointer {
		t = t.Elem()
		v = v.Elem()
	}

	if t.Kind() == reflect.Interface {
		return loadTensors(b, v.Interface(), tensorPath...)
	}

	for i := range t.NumField() {
		f := v.Field(i)
		fullTensorPath := tensorPath
		if tag := t.Field(i).Tag.Get("ggml"); tag != "" {
			tensorName, _, _ := strings.Cut(tag, ",")
			fullTensorPath = append(tensorPath, tensorName)
		}

		if !f.CanSet() {
			continue
		}

		if f.Kind() == reflect.Ptr && f.IsNil() {
			f.Set(reflect.New(f.Type().Elem()))
		} else if f.Kind() == reflect.Interface && f.IsNil() && f.Type().Implements(mlTensorType) {
			if tensor := b.Get(strings.Join(fullTensorPath, ".")); tensor != nil {
				f.Set(reflect.ValueOf(tensor))
				slog.Debug("loaded tensor", "kind", f.Elem().Type(), "", f.Interface())
			}
		}

		if r := reflect.Indirect(f); r.Kind() == reflect.Struct {
			if err := loadTensors(b, f.Interface(), fullTensorPath...); err != nil {
				return err
			}
		} else if r.Kind() == reflect.Slice {
			for i := range r.Len() {
				if err := loadTensors(b, f.Index(i).Addr().Interface(), append(fullTensorPath, strconv.Itoa(i))...); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func Forward(m Model, optsFuncs ...OptionsFunc) (ml.Tensor, error) {
	var opts Options
	for _, optsFunc := range optsFuncs {
		optsFunc(m, &opts)
	}

	ctx := m.Backend().NewContext()
	defer ctx.Close()

	err := opts.Cache.StartForward(ctx, opts.sequences)
	if err != nil {
		return nil, err
	}

	t, err := m.Forward(ctx, opts)
	if err != nil {
		return nil, err
	}

	return ctx.Compute(t), nil
}
