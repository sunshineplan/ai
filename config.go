package ai

import "golang.org/x/time/rate"

type Config struct {
	LLMs   LLMs
	APIKey string

	BaseURL string
	Model   string
	Proxy   string

	Count       *int32
	MaxTokens   *int32
	Temperature *float32
	TopP        *float32

	Limit *rate.Limit
}
