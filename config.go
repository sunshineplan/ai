package ai

type ClientConfig struct {
	LLMs LLMs

	APIKey   string
	Endpoint string
	Proxy    string

	Limit *int64

	Model       string
	ModelConfig ModelConfig
}

type ModelConfig struct {
	Count        *int64
	MaxTokens    *int64
	Temperature  *float64
	TopP         *float64
	JSONResponse *bool
	JSONSchema   *JSONSchema
	Tools        []Function
	ToolConfig   FunctionCallingMode
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
		ai.SetJSONResponse(*cfg.JSONResponse, cfg.JSONSchema)
	}
	ai.SetFunctionCall(cfg.Tools, cfg.ToolConfig)
}

type ClientOption interface {
	Apply(*ClientConfig)
}

func WithAPIKey(apiKey string) ClientOption           { return withAPIKey(apiKey) }
func WithEndpoint(endpoint string) ClientOption       { return withEndpoint(endpoint) }
func WithProxy(proxy string) ClientOption             { return withProxy(proxy) }
func WithLimit(rpm int64) ClientOption                { return withLimit(rpm) }
func WithModel(model string) ClientOption             { return withModel(model) }
func WithModelConfig(config ModelConfig) ClientOption { return withModelConfig(config) }

type withAPIKey string

func (w withAPIKey) Apply(cfg *ClientConfig) { cfg.APIKey = string(w) }

type withEndpoint string

func (w withEndpoint) Apply(cfg *ClientConfig) { cfg.Endpoint = string(w) }

type withProxy string

func (w withProxy) Apply(cfg *ClientConfig) { cfg.Proxy = string(w) }

type withLimit int64

func (w withLimit) Apply(cfg *ClientConfig) { cfg.Limit = (*int64)(&w) }

type withModel string

func (w withModel) Apply(cfg *ClientConfig) { cfg.Model = string(w) }

type withModelConfig ModelConfig

func (w withModelConfig) Apply(cfg *ClientConfig) { cfg.ModelConfig = ModelConfig(w) }
