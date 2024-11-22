package ai

import (
	"encoding"
	"errors"
	"strings"
)

const (
	ChatGPT   LLMs = "ChatGPT"
	Gemini    LLMs = "Gemini"
	Anthropic LLMs = "Anthropic"
)

var llms = []LLMs{ChatGPT, Gemini, Anthropic}

var (
	_ encoding.TextMarshaler   = LLMs("")
	_ encoding.TextUnmarshaler = new(LLMs)
)

type LLMs string

func (m LLMs) MarshalText() ([]byte, error) {
	return []byte(m), nil
}

func (m *LLMs) UnmarshalText(text []byte) error {
	if string(text) == "" {
		return errors.New("empty LLMs name")
	}
	for _, i := range llms {
		if strings.EqualFold(string(i), string(text)) {
			*m = i
			return nil
		}
	}
	return errors.New("unsupported LLMs: " + string(text))
}
