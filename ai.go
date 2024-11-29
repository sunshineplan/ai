package ai

import (
	"context"
	"errors"
)

var ErrAIClosed = errors.New("AI client is nil or already closed")

type AI interface {
	LLMs() LLMs
	Model(context.Context) (string, error)

	Limiter

	SetModel(string)
	Model

	Chatbot
	ChatSession() ChatSession

	Close() error
}

type Function struct {
	Name        string
	Description string
	Parameters  Schema
}

type Schema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Enum       []string       `json:"enum,omitempty"`
	Required   []string       `json:"required"`
}

type FunctionCallingMode int

const (
	FunctionCallingAuto FunctionCallingMode = iota
	FunctionCallingAny
	FunctionCallingNone
)

type Model interface {
	SetFunctionCall([]Function, FunctionCallingMode)
	SetCount(x int64)
	SetMaxTokens(x int64)
	SetTemperature(x float64)
	SetTopP(x float64)
	SetJSONResponse(b bool)
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
	FunctionCalls() []FunctionCall
	TokenCount() TokenCount
}
