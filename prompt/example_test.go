package prompt

import "testing"

func TestExample(t *testing.T) {
	for i, tc := range []struct {
		ex     Example
		prefix string
		output string
	}{
		{Example{nil, "result"}, "%d|", ""},
		{Example{[]string{"abc"}, "result"}, "%d|", "Input:\"\"\"\n1|abc\n\"\"\"\nOutput: result"},
		{Example{[]string{"abc", "def", "ghi"}, "result"}, "", "Input:\"\"\"\nabc\ndef\nghi\n\"\"\"\nOutput: result"},
		{Example{[]string{"abc", "def", "ghi"}, "result"}, "%d|", "Input:\"\"\"\n1|abc\n2|def\n3|ghi\n\"\"\"\nOutput: result"},
	} {
		if output := tc.ex.Sprint(tc.prefix); output != tc.output {
			t.Errorf("#%d: expected %q; got %q", i, tc.output, output)
		}
	}
}
