package prompt

import (
	"context"
	"fmt"
	"math"
	"strings"
	"text/template"
	"time"

	"github.com/sunshineplan/ai"
	"github.com/sunshineplan/workers"
)

const defaultTimeout = 3 * time.Minute

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

	d time.Duration
}

func New(prompt string) *Prompt {
	p := &Prompt{prompt: prompt, d: defaultTimeout}
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
	Tokens int
	Error  error
}

func newWorkers(ai ai.AI) *workers.Workers {
	if rpm := ai.Limit(); rpm != math.MaxInt64 {
		return workers.NewWorkers(rpm)
	}
	return workers.NewWorkers(0)
}

func (prompt *Prompt) Execute(ai ai.AI, input []string, prefix string) (<-chan *Result, int, error) {
	prompts, err := prompt.Prompts(input, prefix)
	if err != nil {
		return nil, 0, err
	}
	n := len(prompts)
	c := make(chan *Result, n)
	go func() {
		newWorkers(ai).Run(context.Background(), workers.SliceJob(prompts, func(i int, p string) {
			resp, err := chat(ai, prompt.d, p)
			if err != nil {
				c <- &Result{i, p, nil, 0, err}
			} else {
				c <- &Result{i, p, resp.Results(), resp.TokenCount().Total, nil}
			}
		}))
		close(c)
	}()
	return c, n, nil
}

func (prompt *Prompt) JobList(ctx context.Context, ai ai.AI, input []string, prefix string, c chan<- *Result) (
	*workers.JobList[*Result], int, error) {
	prompts, err := prompt.Prompts(input, prefix)
	if err != nil {
		return nil, 0, err
	}
	jobList := workers.NewJobList(newWorkers(ai), func(r *Result) {
		resp, err := chat(ai, prompt.d, r.Prompt)
		if err != nil {
			r.Result = nil
			r.Tokens = 0
			r.Error = err
		} else {
			r.Result = resp.Results()
			r.Tokens = resp.TokenCount().Total
			r.Error = nil
		}
		c <- r
	})
	jobList.Start(ctx)
	for i, p := range prompts {
		jobList.PushBack(&Result{Index: i, Prompt: p})
	}
	return jobList, len(prompts), nil
}

func chat(ai ai.AI, d time.Duration, p string) (ai.ChatResponse, error) {
	var ctx context.Context
	var cancel context.CancelFunc
	if d > 0 {
		ctx, cancel = context.WithTimeout(context.Background(), d)
	} else {
		ctx, cancel = context.WithCancel(context.Background())
	}
	defer cancel()
	return ai.Chat(ctx, p)
}
