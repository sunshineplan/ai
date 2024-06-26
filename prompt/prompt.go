package prompt

import (
	"context"
	"fmt"
	"strings"
	"text/template"
	"time"

	"github.com/sunshineplan/ai"
	"github.com/sunshineplan/utils/workers"
)

const (
	defaultTimeout = time.Minute
	defaultWorkers = 3
)

const defaultTemplate = `{{.Request}}{{if .Example}}
###
Example:
{{.Example.Sprint .Prefix}}
###{{end}}{{if .Input}}
Input:"""
{{printBatch .Input .Prefix .Start}}"""{{end}}
Output:`

func printBatch(s []string, prefix string, start int) string {
	var b strings.Builder
	for i, s := range s {
		if prefix == "" {
			fmt.Fprintln(&b, s)
		} else if strings.Count(prefix, "%d") == 0 {
			fmt.Fprintln(&b, prefix+s)
		} else {
			fmt.Fprintln(&b, fmt.Sprintf(prefix, start+i+1)+s)
		}
	}
	return b.String()
}

var defaultFuncMap = template.FuncMap{
	"printBatch": printBatch,
}

type Prompt struct {
	prompt string
	t      *template.Template
	ex     *Example
	n      int

	d       time.Duration
	workers int
}

func New(prompt string) *Prompt {
	p := &Prompt{prompt: prompt, d: defaultTimeout, workers: defaultWorkers}
	p.t = template.Must(template.New("prompt").Funcs(defaultFuncMap).Parse(defaultTemplate))
	return p
}

func (prompt *Prompt) SetTemplate(t *template.Template) *Prompt {
	prompt.t = t
	return prompt
}

func (prompt *Prompt) SetExample(ex Example) *Prompt {
	prompt.ex = &ex
	return prompt
}

func (prompt *Prompt) SetInputN(n int) *Prompt {
	prompt.n = n
	return prompt
}

func (prompt *Prompt) SetAITimeout(d time.Duration) *Prompt {
	prompt.d = d
	return prompt
}

func (prompt *Prompt) SetWorkers(n int) *Prompt {
	prompt.workers = n
	return prompt
}

func (prompt *Prompt) Prompts(input []string, prefix string) (prompts []string, err error) {
	length := len(input)
	if length == 0 {
		return
	}
	n := prompt.n
	if n == 0 {
		n = length
	}
	for i := 0; i < length; i = i + n {
		var s []string
		if i+n < length {
			s = input[i : i+n]
		} else {
			s = input[i:]
		}
		var b strings.Builder
		if err = prompt.t.Execute(&b, struct {
			Request string
			Example *Example
			Input   []string
			Prefix  string
			Start   int
		}{prompt.prompt, prompt.ex, s, prefix, i}); err != nil {
			return nil, err
		}
		prompts = append(prompts, b.String())
	}
	return
}

type Result struct {
	Index  int
	Prompt string
	Result []string
	Error  error
}

func (prompt *Prompt) Execute(ai ai.AI, input []string, prefix string) (<-chan Result, int, error) {
	prompts, err := prompt.Prompts(input, prefix)
	if err != nil {
		return nil, 0, err
	}
	n := len(prompts)
	c := make(chan Result, n)
	go func() {
		workers.RunSlice(prompt.workers, prompts, func(i int, p string) {
			var ctx context.Context
			var cancel context.CancelFunc
			if prompt.d != 0 {
				ctx, cancel = context.WithTimeout(context.Background(), prompt.d)
			} else {
				ctx, cancel = context.WithCancel(context.Background())
			}
			defer cancel()
			resp, err := ai.Chat(ctx, p)
			if err != nil {
				c <- Result{i, p, nil, err}
			} else {
				c <- Result{i, p, resp.Results(), nil}
			}
		})
		close(c)
	}()
	return c, n, nil
}
