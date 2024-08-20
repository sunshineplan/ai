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

type Model interface {
	SetCount(x int64)
	SetMaxTokens(x int64)
	SetTemperature(x float64)
	SetTopP(x float64)
	SetJSONResponse(b bool)
}

type Chatbot interface {
	Chat(context.Context, ...string) (ChatResponse, error)
	ChatStream(context.Context, ...string) (ChatStream, error)
}

type Message struct {
	Content string
	Role    string
}

type ChatSession interface {
	Chatbot
	History() []Message
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
	Results() []string
	TokenCount() TokenCount
}
