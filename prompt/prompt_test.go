package prompt

import (
	"reflect"
	"testing"
)

func TestPrompt(t *testing.T) {
	for i, tc := range []struct {
		prompt  *Prompt
		input   []string
		prefix  string
		prompts []string
	}{
		{
			New("no example single input"),
			[]string{"test"},
			"",
			[]string{"no example single input\nInput: test\nOutput:"},
		},
		{
			New("has example single input").SetExample(Example{[]string{"abc", "def"}, "example", ""}),
			[]string{"test"},
			"",
			[]string{
				"has example single input\nExample:\nInput:\"\"\"\nabc\ndef\n\"\"\"\nOutput: example\n###\nInput: test\nOutput:",
			},
		},
		{
			New("has example with prefix single input").SetExample(Example{[]string{"abc", "def"}, "example", "%d|"}),
			[]string{"test"},
			"",
			[]string{
				"has example with prefix single input\nExample:\nInput:\"\"\"\n1|abc\n2|def\n\"\"\"\nOutput: example\n###\nInput: test\nOutput:",
			},
		},
		{
			New("no example multiple inputs"),
			[]string{"test1", "test2"},
			"",
			[]string{"no example multiple inputs\nInput:\"\"\"\ntest1\ntest2\n\"\"\"\nOutput:"},
		},
		{
			New("no example multiple inputs with prefix"),
			[]string{"test1", "test2"},
			"%d|",
			[]string{"no example multiple inputs with prefix\nInput:\"\"\"\n1|test1\n2|test2\n\"\"\"\nOutput:"},
		},
		{
			New("test limit").SetExample(Example{[]string{"abc", "def"}, "example", "%d|"}).SetLimit(2),
			[]string{"test1", "test2", "test3", "test4"},
			"%d|",
			[]string{
				"test limit\nExample:\nInput:\"\"\"\n1|abc\n2|def\n\"\"\"\nOutput: example\n###\nInput:\"\"\"\n1|test1\n2|test2\n\"\"\"\nOutput:",
				"test limit\nExample:\nInput:\"\"\"\n1|abc\n2|def\n\"\"\"\nOutput: example\n###\nInput:\"\"\"\n3|test3\n4|test4\n\"\"\"\nOutput:",
			},
		},
	} {
		if prompts, err := tc.prompt.execute(tc.input, tc.prefix); err != nil {
			t.Errorf("#%d: error: %s", i, err)
		} else if !reflect.DeepEqual(prompts, tc.prompts) {
			t.Errorf("#%d: expected %q; got %q", i, tc.prompts, prompts)
		}
	}
}
