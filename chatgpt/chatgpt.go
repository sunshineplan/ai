package chatgpt

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"time"

	"github.com/sunshineplan/ai"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/ssestream"
	"golang.org/x/time/rate"
)

const defaultModel = openai.ChatModelGPT4oMini

var _ ai.AI = new(ChatGPT)

type ChatGPT struct {
	*openai.Client
	model       string
	maxTokens   *int64
	temperature *float64
	topP        *float64
	count       *int64
	json        *bool

	limiter *rate.Limiter
}

func New(opts ...ai.ClientOption) (ai.AI, error) {
	cfg := new(ai.ClientConfig)
	for _, i := range opts {
		i.Apply(cfg)
	}
	options := []option.RequestOption{
		option.WithAPIKey(cfg.APIKey),
	}
	if cfg.Endpoint != "" {
		options = append(options, option.WithBaseURL(cfg.Endpoint))
	}
	if cfg.Proxy != "" {
		u, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, err
		}
		if t, ok := http.DefaultTransport.(*http.Transport); ok {
			t = t.Clone()
			t.Proxy = http.ProxyURL(u)
			options = append(options, option.WithHTTPClient(&http.Client{Transport: t}))
		}
	}
	c := NewWithClient(openai.NewClient(options...), cfg.Model)
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

func (chatgpt *ChatGPT) SetLimit(rpm int64) {
	chatgpt.limiter = ai.NewLimiter(rpm)
}

func (chatgpt *ChatGPT) Limit() (rpm int64) {
	if chatgpt.limiter == nil {
		return math.MaxInt64
	}
	return int64(chatgpt.limiter.Limit() / rate.Every(time.Minute))
}

func (ai *ChatGPT) wait(ctx context.Context) error {
	if ai.limiter != nil {
		return ai.limiter.Wait(ctx)
	}
	return nil
}

func (ai *ChatGPT) SetModel(model string)    { ai.model = model }
func (ai *ChatGPT) SetCount(i int64)         { ai.count = &i }
func (ai *ChatGPT) SetMaxTokens(i int64)     { ai.maxTokens = &i }
func (ai *ChatGPT) SetTemperature(f float64) { ai.temperature = &f }
func (ai *ChatGPT) SetTopP(f float64)        { ai.topP = &f }
func (ai *ChatGPT) SetJSONResponse(b bool)   { ai.json = &b }

var _ ai.ChatResponse = new(ChatResponse[*openai.ChatCompletion])

type ChatCompletionResponse interface {
	*openai.ChatCompletion | openai.ChatCompletionChunk
}

type ChatResponse[Response ChatCompletionResponse] struct {
	resp Response
}

func (resp *ChatResponse[Response]) Results() (res []string) {
	switch v := any(resp.resp).(type) {
	case *openai.ChatCompletion:
		for _, i := range v.Choices {
			res = append(res, i.Message.Content)
		}
	case openai.ChatCompletionChunk:
		for _, i := range v.Choices {
			res = append(res, i.Delta.Content)
		}
	}
	return
}

func (resp *ChatResponse[Response]) TokenCount() (res ai.TokenCount) {
	switch v := any(resp.resp).(type) {
	case *openai.ChatCompletion:
		res.Prompt = v.Usage.PromptTokens
		res.Result = v.Usage.CompletionTokens
		res.Total = v.Usage.TotalTokens
	case openai.ChatCompletionChunk:
		res.Prompt = v.Usage.PromptTokens
		res.Result = v.Usage.CompletionTokens
		res.Total = v.Usage.TotalTokens
	}
	return
}

func (resp *ChatResponse[Response]) String() string {
	if res := resp.Results(); len(res) > 0 {
		return res[0]
	}
	return ""
}

func toImagePart(img ai.Image) openai.ChatCompletionContentPartImageParam {
	return openai.ImagePart(fmt.Sprintf("data:%s;base64,%s", img.MIMEType, base64.StdEncoding.EncodeToString(img.Data)))
}

func fromImagePart(img openai.ChatCompletionContentPartImageParam) ai.Image {
	res := regexp.MustCompile("^data:(.+);base64,(.+)$").FindStringSubmatch(img.ImageURL.Value.URL.Value)
	b, err := base64.StdEncoding.DecodeString(res[1])
	if err != nil {
		panic(err)
	}
	return ai.ImageData(res[0], b)
}

func (c *ChatGPT) createRequest(
	one bool,
	history []openai.ChatCompletionMessageParamUnion,
	messages ...ai.Part,
) (req openai.ChatCompletionNewParams) {
	req.Model = openai.String(c.model)
	if c.maxTokens != nil {
		req.MaxTokens = openai.Int(*c.maxTokens)
	}
	if !one && c.count != nil {
		req.N = openai.Int(*c.count)
	}
	if c.temperature != nil {
		req.Temperature = openai.Float(*c.temperature)
	}
	if c.topP != nil {
		req.TopP = openai.Float(*c.topP)
	}
	if c.json != nil && *c.json {
		req.ResponseFormat = openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
			openai.ChatCompletionNewParamsResponseFormat{
				Type: openai.F(openai.ChatCompletionNewParamsResponseFormatTypeJSONObject),
			},
		)
	}
	var msgs []openai.ChatCompletionMessageParamUnion
	msgs = append(msgs, history...)
	for _, i := range messages {
		switch v := i.(type) {
		case ai.Text:
			msgs = append(msgs, openai.UserMessage(string(v)))
		case ai.Image:
			msgs = append(msgs, openai.UserMessageParts(toImagePart((v))))
		}
	}
	req.Messages = openai.F(msgs)
	return
}

func (chatgpt *ChatGPT) chat(
	ctx context.Context,
	session bool,
	history []openai.ChatCompletionMessageParamUnion,
	messages ...ai.Part,
) (resp *openai.ChatCompletion, err error) {
	if chatgpt.Client == nil {
		err = ai.ErrAIClosed
		return
	}
	if err = chatgpt.wait(ctx); err != nil {
		return
	}
	return chatgpt.Client.Chat.Completions.New(ctx, chatgpt.createRequest(session, history, messages...))
}

func (ai *ChatGPT) Chat(ctx context.Context, messages ...ai.Part) (ai.ChatResponse, error) {
	resp, err := ai.chat(ctx, false, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse[*openai.ChatCompletion]{resp}, nil
}

var _ ai.ChatStream = new(ChatStream)

type ChatStream struct {
	stream  *ssestream.Stream[openai.ChatCompletionChunk]
	session *ChatSession
	merged  string
}

func (cs *ChatStream) Next() (ai.ChatResponse, error) {
	if cs.stream.Next() {
		resp := cs.stream.Current()
		if cs.session != nil && len(resp.Choices) > 0 {
			cs.merged += resp.Choices[0].Delta.Content
		}
		return &ChatResponse[openai.ChatCompletionChunk]{resp}, nil
	}
	if err := cs.stream.Err(); err != nil {
		cs.merged = ""
		return nil, err
	}
	if cs.session != nil {
		cs.session.history = append(cs.session.history, openai.ChatCompletionMessage{
			Content: cs.merged,
			Role:    openai.ChatCompletionMessageRoleAssistant,
		})
	}
	return nil, io.EOF
}

func (cs *ChatStream) Close() error {
	return cs.stream.Close()
}

func (chatgpt *ChatGPT) chatStream(
	ctx context.Context,
	history []openai.ChatCompletionMessageParamUnion,
	messages ...ai.Part,
) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if chatgpt.Client == nil {
		return nil, ai.ErrAIClosed
	}
	if err := chatgpt.wait(ctx); err != nil {
		return nil, err
	}
	return chatgpt.Client.Chat.Completions.NewStreaming(ctx, chatgpt.createRequest(true, history, messages...)), nil
}

func (ai *ChatGPT) ChatStream(ctx context.Context, messages ...ai.Part) (ai.ChatStream, error) {
	stream, err := ai.chatStream(ctx, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatStream{stream, nil, ""}, nil
}

var _ ai.ChatSession = new(ChatSession)

type ChatSession struct {
	ai      *ChatGPT
	history []openai.ChatCompletionMessageParamUnion
}

func (session *ChatSession) addUserHistory(messages ...ai.Part) {
	for _, i := range messages {
		switch v := i.(type) {
		case ai.Text:
			session.history = append(session.history, openai.UserMessage(string(v)))
		case ai.Image:
			session.history = append(session.history, openai.UserMessageParts(toImagePart(v)))
		}
	}
}

func (session *ChatSession) Chat(ctx context.Context, messages ...ai.Part) (ai.ChatResponse, error) {
	resp, err := session.ai.chat(ctx, true, session.history, messages...)
	if err != nil {
		return nil, err
	}
	session.addUserHistory(messages...)
	if len(resp.Choices) > 0 {
		session.history = append(session.history, resp.Choices[0].Message)
	}
	return &ChatResponse[*openai.ChatCompletion]{resp}, nil
}

func (session *ChatSession) ChatStream(ctx context.Context, messages ...ai.Part) (ai.ChatStream, error) {
	stream, err := session.ai.chatStream(ctx, session.history, messages...)
	if err != nil {
		return nil, err
	}
	session.addUserHistory(messages...)
	return &ChatStream{stream, session, ""}, nil
}

func (session *ChatSession) History() (history []ai.Content) {
	for _, i := range session.history {
		switch v := i.(type) {
		case openai.ChatCompletionMessage:
			history = append(history, ai.Content{Role: string(v.Role), Parts: []ai.Part{ai.Text(v.Content)}})
		case openai.ChatCompletionUserMessageParam:
			var parts []ai.Part
			for _, i := range v.Content.Value {
				switch v := i.(type) {
				case openai.ChatCompletionContentPartTextParam:
					parts = append(parts, ai.Text(v.Text.Value))
				case openai.ChatCompletionContentPartImageParam:
					parts = append(parts, fromImagePart(v))
				}
			}
			history = append(history, ai.Content{Role: "user", Parts: parts})
		}
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
