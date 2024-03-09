package ai

import (
	"context"
	"net/http"
	"net/url"
)

type AI interface {
	Limiter

	SetModel(string)
	Model

	Chatbot
	ChatSession() Chatbot
}

type Model interface {
	SetCount(x int32)
	SetMaxTokens(x int32)
	SetTemperature(x float32)
	SetTopP(x float32)
}

type Chatbot interface {
	Chat(context.Context, ...string) (ChatResponse, error)
	ChatStream(context.Context, ...string) (ChatStream, error)
}

type ChatStream interface {
	Next() (ChatResponse, error)
}

type ChatResponse interface {
	Results() []string
}

func SetProxy(proxy string) error {
	u, err := url.Parse(proxy)
	if err != nil {
		return err
	}
	http.DefaultTransport.(*http.Transport).Proxy = http.ProxyURL(u)
	return nil
}
