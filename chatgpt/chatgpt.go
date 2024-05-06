package chatgpt

import (
	"context"
	"io"
	"net/http"
	"net/url"

	"github.com/sunshineplan/ai"

	"github.com/sashabaranov/go-openai"
	"golang.org/x/time/rate"
)

const defaultModel = openai.GPT3Dot5Turbo

var _ ai.AI = new(ChatGPT)

type ChatGPT struct {
	*openai.Client
	model       string
	maxTokens   *int32
	temperature *float32
	topP        *float32
	count       *int32
	json        *bool

	limiter *rate.Limiter
}

func New(opts ...ai.ClientOption) (ai.AI, error) {
	cfg := new(ai.ClientConfig)
	for _, i := range opts {
		i.Apply(cfg)
	}
	config := openai.DefaultConfig(cfg.APIKey)
	if cfg.Endpoint != "" {
		config.BaseURL = cfg.Endpoint
	}
	if cfg.Proxy != "" {
		u, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, err
		}
		if t, ok := http.DefaultTransport.(*http.Transport); ok {
			t = t.Clone()
			t.Proxy = http.ProxyURL(u)
			config.HTTPClient = &http.Client{Transport: t}
		}
	}
	c := NewWithClient(openai.NewClientWithConfig(config), cfg.Model)
	if cfg.Limit != nil {
		c.SetLimit(*cfg.Limit)
	}
	ai.ApplyModelConfig(c, cfg.ModelConfig)
	return c, nil
}

func NewWithClient(client *openai.Client, model string) ai.AI {
	if client == nil {
		panic("cannot create AI from nil client")
	}
	if model == "" {
		model = defaultModel
	}
	return &ChatGPT{Client: client, model: model}
}

func (ChatGPT) LLMs() ai.LLMs {
	return ai.ChatGPT
}

func (chatgpt *ChatGPT) Model(_ context.Context) (string, error) {
	return chatgpt.model, nil
}

func (chatgpt *ChatGPT) SetLimit(limit rate.Limit) {
	chatgpt.limiter = ai.NewLimiter(limit)
}

func (ai *ChatGPT) wait(ctx context.Context) error {
	if ai.limiter != nil {
		return ai.limiter.Wait(ctx)
	}
	return nil
}

func (ai *ChatGPT) SetModel(model string)    { ai.model = model }
func (ai *ChatGPT) SetCount(i int32)         { ai.count = &i }
func (ai *ChatGPT) SetMaxTokens(i int32)     { ai.maxTokens = &i }
func (ai *ChatGPT) SetTemperature(f float32) { ai.temperature = &f }
func (ai *ChatGPT) SetTopP(f float32)        { ai.topP = &f }
func (ai *ChatGPT) SetJSONResponse(b bool)   { ai.json = &b }

type ChatGPTResponse interface {
	openai.ChatCompletionResponse | openai.ChatCompletionStreamResponse
}

var _ ai.ChatResponse = new(ChatResponse[openai.ChatCompletionResponse])

type ChatResponse[Response ChatGPTResponse] struct {
	resp Response
}

func (resp *ChatResponse[Response]) Results() (res []string) {
	switch v := any(resp.resp).(type) {
	case openai.ChatCompletionResponse:
		for _, i := range v.Choices {
			res = append(res, i.Message.Content)
		}
	case openai.ChatCompletionStreamResponse:
		for _, i := range v.Choices {
			res = append(res, i.Delta.Content)
		}
	}
	return
}

func (resp *ChatResponse[Response]) String() string {
	if res := resp.Results(); len(res) > 0 {
		return res[0]
	}
	return ""
}

func (ai *ChatGPT) createRequest(
	one bool,
	history []openai.ChatCompletionMessage,
	messages ...string,
) (req openai.ChatCompletionRequest) {
	req.Model = ai.model
	if ai.maxTokens != nil {
		req.MaxTokens = int(*ai.maxTokens)
	}
	if !one && ai.count != nil {
		req.N = int(*ai.count)
	}
	if ai.temperature != nil {
		req.Temperature = *ai.temperature
	}
	if ai.topP != nil {
		req.TopP = *ai.topP
	}
	if ai.json != nil && *ai.json {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	}
	req.Messages = history
	for _, i := range messages {
		req.Messages = append(
			req.Messages,
			openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: i},
		)
	}
	return
}

func (chatgpt *ChatGPT) chat(
	ctx context.Context,
	session bool,
	history []openai.ChatCompletionMessage,
	messages ...string,
) (resp openai.ChatCompletionResponse, err error) {
	if chatgpt.Client == nil {
		err = ai.ErrAIClosed
		return
	}
	if err = chatgpt.wait(ctx); err != nil {
		return
	}
	return chatgpt.CreateChatCompletion(ctx, chatgpt.createRequest(session, history, messages...))
}

func (ai *ChatGPT) Chat(ctx context.Context, messages ...string) (ai.ChatResponse, error) {
	resp, err := ai.chat(ctx, false, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse[openai.ChatCompletionResponse]{resp}, nil
}

var _ ai.ChatStream = new(ChatStream)

type ChatStream struct {
	sr     *openai.ChatCompletionStream
	cs     *ChatSession
	merged string
}

func (stream *ChatStream) Next() (ai.ChatResponse, error) {
	resp, err := stream.sr.Recv()
	if err != nil {
		if err == io.EOF {
			if stream.cs != nil {
				stream.cs.history = append(stream.cs.history, openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleAssistant, Content: stream.merged})
			}
		}
		stream.merged = ""
		return nil, err
	}
	if stream.cs != nil {
		stream.merged += resp.Choices[0].Delta.Content
	}
	return &ChatResponse[openai.ChatCompletionStreamResponse]{resp}, nil
}

func (stream *ChatStream) Close() error {
	return stream.sr.Close()
}

func (chatgpt *ChatGPT) chatStream(
	ctx context.Context,
	history []openai.ChatCompletionMessage,
	messages ...string,
) (*openai.ChatCompletionStream, error) {
	if chatgpt.Client == nil {
		return nil, ai.ErrAIClosed
	}
	if err := chatgpt.wait(ctx); err != nil {
		return nil, err
	}
	req := chatgpt.createRequest(true, history, messages...)
	req.Stream = true
	return chatgpt.CreateChatCompletionStream(ctx, req)
}

func (ai *ChatGPT) ChatStream(ctx context.Context, messages ...string) (ai.ChatStream, error) {
	stream, err := ai.chatStream(ctx, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatStream{stream, nil, ""}, nil
}

var _ ai.ChatSession = new(ChatSession)

type ChatSession struct {
	ai      *ChatGPT
	history []openai.ChatCompletionMessage
}

func addToHistory(history *[]openai.ChatCompletionMessage, role string, messages ...string) {
	for _, i := range messages {
		*history = append(
			*history,
			openai.ChatCompletionMessage{Role: role, Content: i},
		)
	}
}

func (session *ChatSession) Chat(ctx context.Context, messages ...string) (ai.ChatResponse, error) {
	resp, err := session.ai.chat(ctx, true, session.history, messages...)
	if err != nil {
		return nil, err
	}
	addToHistory(&session.history, openai.ChatMessageRoleUser, messages...)
	session.history = append(session.history, resp.Choices[0].Message)
	return &ChatResponse[openai.ChatCompletionResponse]{resp}, nil
}

func (session *ChatSession) ChatStream(ctx context.Context, messages ...string) (ai.ChatStream, error) {
	stream, err := session.ai.chatStream(ctx, session.history, messages...)
	if err != nil {
		return nil, err
	}
	addToHistory(&session.history, openai.ChatMessageRoleUser, messages...)
	return &ChatStream{stream, session, ""}, nil
}

func (session *ChatSession) History() (history []ai.Message) {
	for _, i := range session.history {
		history = append(history, ai.Message{Content: i.Content, Role: i.Role})
	}
	return
}

func (ai *ChatGPT) ChatSession() ai.ChatSession {
	return &ChatSession{ai: ai}
}

func (ai *ChatGPT) Close() error {
	ai.Client = nil
	return nil
}
