package chatgpt

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sunshineplan/ai"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/openai/openai-go/shared"
	"golang.org/x/time/rate"
)

const defaultModel = openai.ChatModelGPT4oMini

var _ ai.AI = new(ChatGPT)

type ChatGPT struct {
	*openai.Client
	model       string
	toolChoice  openai.ChatCompletionToolChoiceOptionUnionParam
	tools       []openai.ChatCompletionToolParam
	maxTokens   *int64
	temperature *float64
	topP        *float64
	count       *int64
	json        openai.ChatCompletionNewParamsResponseFormatUnion

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

func (ai *ChatGPT) SetModel(model string) { ai.model = model }
func (chatgpt *ChatGPT) SetFunctionCall(f []ai.Function, mode ai.FunctionCallingMode) {
	if chatgpt.tools = nil; len(f) == 0 {
		chatgpt.toolChoice = nil
		return
	}
	for _, i := range f {
		var parameters shared.FunctionParameters
		b, _ := json.Marshal(i.Parameters)
		_ = json.Unmarshal(b, &parameters)
		chatgpt.tools = append(chatgpt.tools, openai.ChatCompletionToolParam{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(shared.FunctionDefinitionParam{
				Name:        openai.String(i.Name),
				Description: openai.String(i.Description),
				Parameters:  openai.F(parameters),
			}),
		})
	}
	switch mode {
	case ai.FunctionCallingAuto:
		chatgpt.toolChoice = openai.ChatCompletionToolChoiceOptionAutoAuto
	case ai.FunctionCallingAny:
		chatgpt.toolChoice = openai.ChatCompletionToolChoiceOptionAutoRequired
	case ai.FunctionCallingNone:
		chatgpt.toolChoice = openai.ChatCompletionToolChoiceOptionAutoNone
	default:
		chatgpt.toolChoice = nil
	}
}
func (ai *ChatGPT) SetCount(i int64)         { ai.count = &i }
func (ai *ChatGPT) SetMaxTokens(i int64)     { ai.maxTokens = &i }
func (ai *ChatGPT) SetTemperature(f float64) { ai.temperature = &f }
func (ai *ChatGPT) SetTopP(f float64)        { ai.topP = &f }
func (ai *ChatGPT) SetJSONResponse(set bool, schema *ai.JSONSchema) {
	var responseFormat openai.ChatCompletionNewParamsResponseFormatUnion
	if set {
		if schema != nil {
			var format any
			b, _ := json.Marshal(schema.Schema)
			_ = json.Unmarshal(b, &format)
			responseFormat = shared.ResponseFormatJSONSchemaParam{
				Type: openai.F(shared.ResponseFormatJSONSchemaTypeJSONSchema),
				JSONSchema: openai.F(shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:        openai.String(schema.Name),
					Description: openai.String(schema.Description),
					Schema:      openai.F(format),
				}),
			}
		} else {
			responseFormat = shared.ResponseFormatJSONObjectParam{
				Type: openai.F(shared.ResponseFormatJSONObjectTypeJSONObject),
			}
		}
	}
	ai.json = responseFormat
}

var _ ai.ChatResponse = new(ChatResponse[*openai.ChatCompletion])

type ChatCompletionResponse interface {
	*openai.ChatCompletion | openai.ChatCompletionChunk
}

type ChatResponse[Response ChatCompletionResponse] struct {
	resp Response
}

func (resp *ChatResponse[Response]) Raw() any {
	return resp.resp
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

func (resp *ChatResponse[Response]) FunctionCalls() (res []ai.FunctionCall) {
	switch v := any(resp.resp).(type) {
	case *openai.ChatCompletion:
		for _, i := range v.Choices {
			for _, i := range i.Message.ToolCalls {
				res = append(res, ai.FunctionCall{ID: i.ID, Name: i.Function.Name, Arguments: i.Function.Arguments})
			}
		}
	case openai.ChatCompletionChunk:
		for _, i := range v.Choices {
			for _, i := range i.Delta.ToolCalls {
				res = append(res, ai.FunctionCall{ID: i.ID, Name: i.Function.Name, Arguments: i.Function.Arguments})
			}
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
	if res := resp.FunctionCalls(); len(res) > 0 {
		var args []string
		for _, i := range res {
			args = append(args, i.Arguments)
		}
		return strings.Join(args, "\n")
	}
	return ""
}

func toImagePart(img ai.Image) openai.ChatCompletionContentPartImageParam {
	return openai.ImagePart(string(img))
}

func fromImagePart(img openai.ChatCompletionContentPartImageParam) ai.Image {
	return ai.Image(img.ImageURL.Value.URL.Value)
}

func (c *ChatGPT) createRequest(
	one bool,
	history []openai.ChatCompletionMessageParamUnion,
	messages ...ai.Part,
) (req openai.ChatCompletionNewParams) {
	req.Model = openai.String(c.model)
	if c.toolChoice != nil {
		req.ToolChoice = openai.F(c.toolChoice)
	}
	if len(c.tools) > 0 {
		req.Tools = openai.F(c.tools)
	}
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
	if c.json != nil {
		req.ResponseFormat = openai.F(c.json)
	}
	var msgs []openai.ChatCompletionMessageParamUnion
	for _, i := range history {
		switch v := i.(type) {
		case openai.ChatCompletionMessage:
			if len(v.ToolCalls) > 0 {
				continue
			}
		}
		msgs = append(msgs, i)
	}
	for _, i := range messages {
		switch v := i.(type) {
		case ai.Text:
			msgs = append(msgs, openai.UserMessage(string(v)))
		case ai.Image:
			msgs = append(msgs, openai.UserMessageParts(toImagePart(v)))
		case ai.Blob:
			msgs = append(msgs, openai.UserMessageParts(toImagePart(ai.ImageData(v.MIMEType, v.Data))))
		case ai.FunctionResponse:
			msgs = append(msgs, openai.ToolMessage(v.ID, v.Response))
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
	stream    *ssestream.Stream[openai.ChatCompletionChunk]
	session   *ChatSession
	content   string
	toolCalls map[string]*openai.ChatCompletionMessageToolCall
}

func (cs *ChatStream) Next() (ai.ChatResponse, error) {
	if cs.stream.Next() {
		resp := cs.stream.Current()
		if cs.session != nil && len(resp.Choices) > 0 {
			cs.content += resp.Choices[0].Delta.Content
			for _, i := range resp.Choices[0].Delta.ToolCalls {
				if cs.toolCalls == nil {
					cs.toolCalls = make(map[string]*openai.ChatCompletionMessageToolCall)
				}
				if tc, ok := cs.toolCalls[i.ID]; ok {
					tc.Function.Name += i.Function.Name
					tc.Function.Arguments += i.Function.Arguments
				} else {
					cs.toolCalls[i.ID] = &openai.ChatCompletionMessageToolCall{
						Type: openai.ChatCompletionMessageToolCallTypeFunction,
						ID:   i.ID,
						Function: openai.ChatCompletionMessageToolCallFunction{
							Name:      i.Function.Name,
							Arguments: i.Function.Arguments,
						},
					}
				}
			}
		}
		return &ChatResponse[openai.ChatCompletionChunk]{resp}, nil
	}
	if err := cs.stream.Err(); err != nil {
		cs.content = ""
		return nil, err
	}
	if cs.session != nil {
		var toolCalls []openai.ChatCompletionMessageToolCall
		for _, i := range cs.toolCalls {
			toolCalls = append(toolCalls, *i)
		}
		if cs.content != "" || len(toolCalls) > 0 {
			cs.session.history = append(cs.session.history, openai.ChatCompletionMessage{
				Content:   cs.content,
				ToolCalls: toolCalls,
				Role:      openai.ChatCompletionMessageRoleAssistant,
			})
		}
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
	return &ChatStream{stream, nil, "", nil}, nil
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
		case ai.Blob:
			session.history = append(session.history, openai.UserMessageParts(toImagePart(ai.ImageData(v.MIMEType, v.Data))))
		case ai.FunctionResponse:
			session.history = append(session.history, openai.ToolMessage(v.ID, v.Response))
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
	return &ChatStream{stream, session, "", nil}, nil
}

func (session *ChatSession) History() (history []ai.Content) {
	for _, i := range session.history {
		switch v := i.(type) {
		case openai.ChatCompletionMessage:
			history = append(history, ai.Content{Role: string(v.Role), Parts: []ai.Part{ai.Text(v.Content)}})
			for _, i := range v.ToolCalls {
				history = append(history, ai.Content{Role: string(v.Role), Parts: []ai.Part{
					ai.FunctionCall{ID: i.ID, Name: i.Function.Name, Arguments: i.Function.Arguments},
				}})
			}
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
		case openai.ChatCompletionToolMessageParam:
			for _, i := range v.Content.Value {
				history = append(history, ai.Content{Role: "tool", Parts: []ai.Part{
					ai.FunctionResponse{ID: v.ToolCallID.Value, Response: i.Text.Value},
				}})
			}
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
