package prompt

import "fmt"

type Example struct {
	Input  []string
	Output string
}

func (ex Example) Sprint(prefix string) string {
	switch len(ex.Input) {
	case 0:
		return ""
	default:
		return fmt.Sprintf("Input:\"\"\"\n%s\"\"\"\nOutput: %s", printBatch(ex.Input, prefix, 0), ex.Output)
	}
}
