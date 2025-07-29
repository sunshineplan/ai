package ai

import (
	"context"
	"encoding"
	"errors"
	"strings"
)

var ErrAIClosed = errors.New("AI client is nil or already closed")

type AI interface {
	LLMs() LLMs
	Model() string

	ListModels(context.Context) ([]string, error)

	Limiter

	SetModel(string)
	Model

	Chatbot
	ChatSession() ChatSession

	Close() error
}

type Schema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties,omitempty"`
	Enum       []string       `json:"enum,omitempty"`
	Items      *Schema        `json:"items,omitempty"`
	Required   []string       `json:"required,omitempty"`
}

type JSONSchema struct {
	Name        string
	Description string
	Schema      Schema
}

type Function struct {
	Name        string
	Description string
	Parameters  Schema
}

var _ encoding.TextUnmarshaler = new(FunctionCallingMode)

type FunctionCallingMode int

func (m *FunctionCallingMode) UnmarshalText(text []byte) error {
	switch strings.ToLower(string(text)) {
	case "auto":
		*m = FunctionCallingAuto
	case "any":
		*m = FunctionCallingAny
	case "none":
		*m = FunctionCallingNone
	default:
		*m = 0
	}
	return nil
}

const (
	FunctionCallingAuto FunctionCallingMode = iota + 1
	FunctionCallingAny
	FunctionCallingNone
)

type Model interface {
	SetFunctionCall([]Function, FunctionCallingMode)
	SetCount(x int64)
	SetMaxTokens(x int64)
	SetTemperature(x float64)
	SetTopP(x float64)
	SetJSONResponse(set bool, schema *JSONSchema)
	SetThinking(set bool)
}

type Chatbot interface {
	Chat(context.Context, ...Part) (ChatResponse, error)
	ChatStream(context.Context, ...Part) (ChatStream, error)
}

type ChatSession interface {
	Chatbot
	History() []Content
}

type ChatStream interface {
	Next() (ChatResponse, error)
	Close() error
}

type TokenCount struct {
	Prompt int64
	Result int64
	Total  int64
}

type ChatResponse interface {
	Raw() any
	Results() []string
	Thoughts() []string
	FunctionCalls() []FunctionCall
	TokenCount() TokenCount
}
