package client

import (
	"context"
	"errors"

	"github.com/sunshineplan/ai"
	"github.com/sunshineplan/ai/chatgpt"
	"github.com/sunshineplan/ai/gemini"
)

func New(cfg ai.ClientConfig) (client ai.AI, err error) {
	if cfg.LLMs == "" {
		return nil, errors.New("empty AI")
	}
	opts := []ai.ClientOption{
		ai.WithAPIKey(cfg.APIKey),
		ai.WithEndpoint(cfg.Endpoint),
		ai.WithProxy(cfg.Proxy),
		ai.WithModel(cfg.Model),
		ai.WithModelConfig(cfg.ModelConfig),
	}
	if cfg.Limit != nil {
		opts = append(opts, ai.WithLimit(*cfg.Limit))
	}
	switch cfg.LLMs {
	case ai.ChatGPT:
		client, err = chatgpt.New(opts...)
	case ai.Gemini:
		client, err = gemini.New(context.Background(), opts...)
	default:
		err = errors.New("unknown LLMs: " + string(cfg.LLMs))
	}
	return
}
