package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sunshineplan/ai"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"golang.org/x/time/rate"
)

const defaultModel = anthropic.ModelClaude3_5SonnetLatest

var DefaultMaxTokens int64 = 1024

var _ ai.AI = new(Anthropic)

type Anthropic struct {
	*anthropic.Client
	model       string
	toolChoice  anthropic.ToolChoiceUnionParam
	tools       []anthropic.ToolParam
	maxTokens   *int64
	temperature *float64
	topP        *float64

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
	c := NewWithClient(anthropic.NewClient(options...), cfg.Model)
	if cfg.Limit != nil {
		c.SetLimit(*cfg.Limit)
	}
	ai.ApplyModelConfig(c, cfg.ModelConfig)
	return c, nil
}

func NewWithClient(client *anthropic.Client, model string) ai.AI {
	if client == nil {
		panic("cannot create AI from nil client")
	}
	if model == "" {
		model = defaultModel
	}
	return &Anthropic{Client: client, model: model}
}

func (Anthropic) LLMs() ai.LLMs {
	return ai.Anthropic
}

func (anthropic *Anthropic) Model(_ context.Context) (string, error) {
	return anthropic.model, nil
}

func (anthropic *Anthropic) SetLimit(rpm int64) {
	anthropic.limiter = ai.NewLimiter(rpm)
}

func (anthropic *Anthropic) Limit() (rpm int64) {
	if anthropic.limiter == nil {
		return math.MaxInt64
	}
	return int64(anthropic.limiter.Limit() / rate.Every(time.Minute))
}

func (ai *Anthropic) wait(ctx context.Context) error {
	if ai.limiter != nil {
		return ai.limiter.Wait(ctx)
	}
	return nil
}

func (ai *Anthropic) SetModel(model string) { ai.model = model }
func (a *Anthropic) SetFunctionCall(f []ai.Function, mode ai.FunctionCallingMode) {
	if a.tools = nil; len(f) == 0 {
		a.toolChoice = nil
		return
	}
	for _, i := range f {
		a.tools = append(a.tools, anthropic.ToolParam{
			Name:        anthropic.String(i.Name),
			Description: anthropic.String(i.Description),
			InputSchema: anthropic.F[any](i.Parameters),
		})
	}
	switch mode {
	case ai.FunctionCallingAuto:
		a.toolChoice = anthropic.ToolChoiceAutoParam{Type: anthropic.F(anthropic.ToolChoiceAutoTypeAuto)}
	case ai.FunctionCallingAny:
		a.toolChoice = anthropic.ToolChoiceAnyParam{Type: anthropic.F(anthropic.ToolChoiceAnyTypeAny)}
	case ai.FunctionCallingNone:
		a.toolChoice = anthropic.ToolChoiceToolParam{
			Type:                   anthropic.F(anthropic.ToolChoiceToolTypeTool),
			Name:                   anthropic.String("none"),
			DisableParallelToolUse: anthropic.Bool(true),
		}
	default:
		a.toolChoice = nil
	}
}
func (ai *Anthropic) SetMaxTokens(i int64)     { ai.maxTokens = &i }
func (ai *Anthropic) SetTemperature(f float64) { ai.temperature = &f }
func (ai *Anthropic) SetTopP(f float64)        { ai.topP = &f }

func (ai *Anthropic) SetCount(i int64) {
	fmt.Println("Anthropic doesn't support SetCount")
}
func (ai *Anthropic) SetJSONResponse(b bool) {
	fmt.Println("Anthropic currently doesn't support SetJSONResponse")
}

var _ ai.ChatResponse = new(ChatResponse[*anthropic.Message])

type ChatCompletionResponse interface {
	*anthropic.Message | anthropic.MessageStreamEvent
}

type ChatResponse[Response ChatCompletionResponse] struct {
	resp Response
}

func (resp *ChatResponse[Response]) Raw() any {
	return resp.resp
}

func (resp *ChatResponse[Response]) Results() (res []string) {
	switch v := any(resp.resp).(type) {
	case *anthropic.Message:
		for _, i := range v.Content {
			if v, ok := i.AsUnion().(anthropic.TextBlock); ok {
				res = append(res, v.Text)
			}
		}
	case anthropic.MessageStreamEvent:
		switch v := v.Delta.(type) {
		case anthropic.MessageDeltaEventDelta:
			// ignored
		case anthropic.ContentBlockDeltaEventDelta:
			if v, ok := v.AsUnion().(anthropic.TextDelta); ok {
				if v.Text != "" {
					res = append(res, v.Text)
				}
			}
		}
	}
	return
}

func (resp *ChatResponse[Response]) FunctionCalls() (res []ai.FunctionCall) {
	switch v := any(resp.resp).(type) {
	case *anthropic.Message:
		for _, i := range v.Content {
			if v, ok := i.AsUnion().(anthropic.ToolUseBlock); ok {
				res = append(res, ai.FunctionCall{ID: v.ID, Name: v.Name, Arguments: string(v.Input)})
			}
		}
	case anthropic.MessageStreamEvent:
		switch v := v.Delta.(type) {
		case anthropic.MessageDeltaEventDelta:
			// ignored
		case anthropic.ContentBlockDeltaEventDelta:
			if v, ok := v.AsUnion().(anthropic.InputJSONDelta); ok {
				if v.PartialJSON != "" {
					res = append(res, ai.FunctionCall{Arguments: v.PartialJSON})
				}
			}
		}
		switch v := v.ContentBlock.(type) {
		case anthropic.ContentBlockStartEventContentBlock:
			if v, ok := v.AsUnion().(anthropic.ToolUseBlock); ok {
				res = append(res, ai.FunctionCall{ID: v.ID, Name: v.Name, Arguments: string(v.Input)})
			}
		}
	}
	return
}

func (resp *ChatResponse[Response]) TokenCount() (res ai.TokenCount) {
	switch v := any(resp.resp).(type) {
	case *anthropic.Message:
		res.Prompt = v.Usage.InputTokens
		res.Result = v.Usage.OutputTokens
	case anthropic.MessageStreamEvent:
		res.Prompt = v.Message.Usage.InputTokens
		res.Result = v.Message.Usage.OutputTokens
	}
	res.Total = res.Prompt + res.Result
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

func toImageBlock(img ai.Image) anthropic.ImageBlockParam {
	mime, data := img.Data()
	return anthropic.NewImageBlockBase64(mime, base64.StdEncoding.EncodeToString(data))
}

func fromImageBlockSource(src anthropic.ImageBlockParamSource) ai.Image {
	b, err := base64.StdEncoding.DecodeString(src.Data.Value)
	if err != nil {
		panic(err)
	}
	return ai.ImageData(src.MediaType.String(), b)
}

func (c *Anthropic) createRequest(
	history []anthropic.MessageParam,
	messages ...ai.Part,
) (req anthropic.MessageNewParams) {
	req.Model = anthropic.String(c.model)
	if c.toolChoice != nil {
		req.ToolChoice = anthropic.F(c.toolChoice)
	}
	if len(c.tools) > 0 {
		req.Tools = anthropic.F(c.tools)
	}
	if c.maxTokens != nil {
		req.MaxTokens = anthropic.Int(*c.maxTokens)
	} else {
		req.MaxTokens = anthropic.Int(DefaultMaxTokens)
	}
	if c.temperature != nil {
		req.Temperature = anthropic.Float(*c.temperature)
	}
	if c.topP != nil {
		req.TopP = anthropic.Float(*c.topP)
	}
	var msgs []anthropic.MessageParam
	for _, i := range history {
		for _, v := range i.Content.Value {
			switch v.(type) {
			case anthropic.ToolUseBlockParam:
				continue
			}
		}
		msgs = append(msgs, i)
	}
	for _, i := range messages {
		switch v := i.(type) {
		case ai.Text:
			msgs = append(msgs, anthropic.NewUserMessage(anthropic.NewTextBlock(string(v))))
		case ai.Image:
			msgs = append(msgs, anthropic.NewUserMessage(toImageBlock(v)))
		case ai.Blob:
			msgs = append(msgs, anthropic.NewUserMessage(toImageBlock(ai.ImageData(v.MIMEType, v.Data))))
		case ai.FunctionResponse:
			msgs = append(msgs, anthropic.NewUserMessage(anthropic.NewToolResultBlock(v.ID, v.Response, false)))
		}
	}
	req.Messages = anthropic.F(msgs)
	return
}

func (anthropic *Anthropic) chat(
	ctx context.Context,
	history []anthropic.MessageParam,
	messages ...ai.Part,
) (resp *anthropic.Message, err error) {
	if anthropic.Client == nil {
		err = ai.ErrAIClosed
		return
	}
	if err = anthropic.wait(ctx); err != nil {
		return
	}
	return anthropic.Client.Messages.New(ctx, anthropic.createRequest(history, messages...))
}

func (ai *Anthropic) Chat(ctx context.Context, messages ...ai.Part) (ai.ChatResponse, error) {
	resp, err := ai.chat(ctx, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse[*anthropic.Message]{resp}, nil
}

var _ ai.ChatStream = new(ChatStream)

type ChatStream struct {
	stream    *ssestream.Stream[anthropic.MessageStreamEvent]
	session   *ChatSession
	content   string
	current   *ai.FunctionCall
	toolCalls map[*ai.FunctionCall]string
}

func (cs *ChatStream) Next() (ai.ChatResponse, error) {
	if cs.stream.Next() {
		resp := cs.stream.Current()
		if cs.session != nil {
			if resp.Delta != nil {
				switch v := resp.Delta.(type) {
				case anthropic.MessageDeltaEventDelta:
					// ignored
				case anthropic.ContentBlockDeltaEventDelta:
					switch v.Type {
					case anthropic.ContentBlockDeltaEventDeltaTypeTextDelta:
						cs.content += v.Text
					case anthropic.ContentBlockDeltaEventDeltaTypeInputJSONDelta:
						if cs.toolCalls == nil {
							cs.toolCalls = make(map[*ai.FunctionCall]string)
						}
						cs.toolCalls[cs.current] += v.PartialJSON
					}
				}
			}
			if resp.ContentBlock != nil {
				switch v := resp.ContentBlock.(type) {
				case anthropic.ContentBlockStartEventContentBlock:
					switch v.Type {
					case anthropic.ContentBlockStartEventContentBlockTypeText:
						cs.content += v.Text
					case anthropic.ContentBlockStartEventContentBlockTypeToolUse:
						cs.current = &ai.FunctionCall{ID: v.ID, Name: v.Name}
					}
				}
			}
		}
		return &ChatResponse[anthropic.MessageStreamEvent]{resp}, nil
	}
	if err := cs.stream.Err(); err != nil {
		cs.content = ""
		return nil, err
	}
	if cs.session != nil {
		if cs.content != "" {
			cs.session.history = append(cs.session.history, anthropic.NewAssistantMessage(anthropic.NewTextBlock(cs.content)))
		}
		if len(cs.toolCalls) > 0 {
			for k, v := range cs.toolCalls {
				cs.session.history = append(
					cs.session.history,
					anthropic.NewAssistantMessage(anthropic.NewToolUseBlockParam(k.ID, k.Name, v)),
				)
			}
		}
	}
	return nil, io.EOF
}

func (cs *ChatStream) Close() error {
	return cs.stream.Close()
}

func (anthropic *Anthropic) chatStream(
	ctx context.Context,
	history []anthropic.MessageParam,
	messages ...ai.Part,
) (*ssestream.Stream[anthropic.MessageStreamEvent], error) {
	if anthropic.Client == nil {
		return nil, ai.ErrAIClosed
	}
	if err := anthropic.wait(ctx); err != nil {
		return nil, err
	}
	return anthropic.Client.Messages.NewStreaming(ctx, anthropic.createRequest(history, messages...)), nil
}

func (ai *Anthropic) ChatStream(ctx context.Context, messages ...ai.Part) (ai.ChatStream, error) {
	stream, err := ai.chatStream(ctx, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatStream{stream, nil, "", nil, nil}, nil
}

var _ ai.ChatSession = new(ChatSession)

type ChatSession struct {
	ai      *Anthropic
	history []anthropic.MessageParam
}

func (session *ChatSession) addUserHistory(messages ...ai.Part) {
	for _, i := range messages {
		switch v := i.(type) {
		case ai.Text:
			session.history = append(session.history, anthropic.NewUserMessage(anthropic.NewTextBlock(string(v))))
		case ai.Image:
			session.history = append(session.history, anthropic.NewUserMessage(toImageBlock(v)))
		case ai.Blob:
			session.history = append(session.history, anthropic.NewUserMessage(toImageBlock(ai.ImageData(v.MIMEType, v.Data))))
		case ai.FunctionResponse:
			session.history = append(session.history, anthropic.NewUserMessage(anthropic.NewToolResultBlock(v.ID, v.Response, false)))
		}
	}
}

func (session *ChatSession) Chat(ctx context.Context, messages ...ai.Part) (ai.ChatResponse, error) {
	resp, err := session.ai.chat(ctx, session.history, messages...)
	if err != nil {
		return nil, err
	}
	session.addUserHistory(messages...)
	session.history = append(session.history, resp.ToParam())
	return &ChatResponse[*anthropic.Message]{resp}, nil
}

func (session *ChatSession) ChatStream(ctx context.Context, messages ...ai.Part) (ai.ChatStream, error) {
	stream, err := session.ai.chatStream(ctx, session.history, messages...)
	if err != nil {
		return nil, err
	}
	session.addUserHistory(messages...)
	return &ChatStream{stream, session, "", nil, nil}, nil
}

func (session *ChatSession) History() (history []ai.Content) {
	for _, i := range session.history {
		for _, v := range i.Content.Value {
			switch v := v.(type) {
			case anthropic.TextBlockParam:
				history = append(history, ai.Content{Role: i.Role.String(), Parts: []ai.Part{ai.Text(v.Text.Value)}})
			case anthropic.ImageBlockParam:
				history = append(history, ai.Content{Role: i.Role.String(), Parts: []ai.Part{fromImageBlockSource(v.Source.Value)}})
			case anthropic.ToolUseBlockParam:
				args, err := json.Marshal(v.Input.Value)
				if err != nil {
					panic(err)
				}
				history = append(history, ai.Content{Role: i.Role.String(), Parts: []ai.Part{ai.FunctionCall{
					ID:        v.ID.Value,
					Name:      v.Name.Value,
					Arguments: string(args),
				}}})
			case anthropic.ToolResultBlockParam:
				for _, ii := range v.Content.Value {
					switch vv := ii.(type) {
					case anthropic.TextBlockParam:
						history = append(history, ai.Content{Role: i.Role.String(), Parts: []ai.Part{ai.FunctionResponse{
							ID:       v.ToolUseID.Value,
							Response: vv.Text.Value,
						}}})
					}
				}
			}
		}
	}
	return
}

func (ai *Anthropic) ChatSession() ai.ChatSession {
	return &ChatSession{ai: ai}
}

func (ai *Anthropic) Close() error {
	ai.Client = nil
	return nil
}
