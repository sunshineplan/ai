package config

import (
	"errors"

	"github.com/sunshineplan/ai"
	"github.com/sunshineplan/ai/chatgpt"
	"github.com/sunshineplan/ai/gemini"
)

func New(cfg ai.Config) (client ai.AI, err error) {
	switch cfg.LLMs {
	case "":
		return nil, errors.New("empty AI")
	case ai.ChatGPT:
		client = chatgpt.NewWithBaseURL(cfg.APIKey, cfg.BaseURL)
	case ai.Gemini:
		client, err = gemini.NewWithEndpoint(cfg.APIKey, cfg.BaseURL)
		if err != nil {
			return
		}
	default:
		return nil, errors.New("unknown LLMs: " + string(cfg.LLMs))
	}
	if cfg.Model != "" {
		client.SetModel(cfg.Model)
	}
	if cfg.Proxy != "" {
		ai.SetProxy(cfg.Proxy)
	}
	if cfg.Count != nil {
		client.SetCount(*cfg.Count)
	}
	if cfg.MaxTokens != nil {
		client.SetMaxTokens(*cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		client.SetTemperature(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		client.SetTopP(*cfg.TopP)
	}
	if cfg.Limit != nil {
		client.SetLimit(*cfg.Limit)
	}
	return
}
