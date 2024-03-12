package prompt

import "fmt"

type Example struct {
	Input  []string
	Output string
	Prefix string
}

func (ex Example) String() string {
	switch len(ex.Input) {
	case 0:
		return ""
	case 1:
		return fmt.Sprintf("Input: %s\nOutput: %s", ex.Input[0], ex.Output)
	default:
		return fmt.Sprintf("Input:%s\nOutput: %s", printBatch(ex.Input, ex.Prefix, 0), ex.Output)
	}
}
