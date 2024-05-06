package ai

import "golang.org/x/time/rate"

type ClientConfig struct {
	LLMs LLMs

	APIKey   string
	Endpoint string
	Proxy    string

	Limit *rate.Limit

	Model       string
	ModelConfig ModelConfig
}

type ModelConfig struct {
	Count        *int32
	MaxTokens    *int32
	Temperature  *float32
	TopP         *float32
	JSONResponse *bool
}

func ApplyModelConfig(ai AI, cfg ModelConfig) {
	if cfg.Count != nil {
		ai.SetCount(*cfg.Count)
	}
	if cfg.MaxTokens != nil {
		ai.SetMaxTokens(*cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		ai.SetTemperature(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		ai.SetTopP(*cfg.TopP)
	}
	if cfg.JSONResponse != nil {
		ai.SetJSONResponse(*cfg.JSONResponse)
	}
}

type ClientOption interface {
	Apply(*ClientConfig)
}

func WithAPIKey(apiKey string) ClientOption           { return withAPIKey(apiKey) }
func WithEndpoint(endpoint string) ClientOption       { return withEndpoint(endpoint) }
func WithProxy(proxy string) ClientOption             { return withProxy(proxy) }
func WithLimit(limit rate.Limit) ClientOption         { return withLimit(limit) }
func WithModel(model string) ClientOption             { return withModel(model) }
func WithModelConfig(config ModelConfig) ClientOption { return withModelConfig(config) }

type withAPIKey string

func (w withAPIKey) Apply(cfg *ClientConfig) { cfg.APIKey = string(w) }

type withEndpoint string

func (w withEndpoint) Apply(cfg *ClientConfig) { cfg.Endpoint = string(w) }

type withProxy string

func (w withProxy) Apply(cfg *ClientConfig) { cfg.Proxy = string(w) }

type withLimit rate.Limit

func (w withLimit) Apply(cfg *ClientConfig) { cfg.Limit = (*rate.Limit)(&w) }

type withModel string

func (w withModel) Apply(cfg *ClientConfig) { cfg.Model = string(w) }

type withModelConfig ModelConfig

func (w withModelConfig) Apply(cfg *ClientConfig) { cfg.ModelConfig = ModelConfig(w) }
