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
	"github.com/anthropics/anthropic-sdk-go/packages/param"
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
	tools       []anthropic.ToolUnionParam
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

func NewWithClient(client anthropic.Client, model string) ai.AI {
	if model == "" {
		model = defaultModel
	}
	return &Anthropic{Client: &client, model: model}
}

func (Anthropic) LLMs() ai.LLMs {
	return ai.Anthropic
}

func (anthropic *Anthropic) Model() string {
	return anthropic.model
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
		a.toolChoice = anthropic.ToolChoiceUnionParam{}
		return
	}
	for _, i := range f {
		a.tools = append(a.tools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        i.Name,
				Description: anthropic.String(i.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: i.Parameters,
				},
			},
		})
	}
	switch mode {
	case ai.FunctionCallingAuto:
		a.toolChoice = anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}
	case ai.FunctionCallingAny:
		a.toolChoice = anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	case ai.FunctionCallingNone:
		a.toolChoice = anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	default:
		a.toolChoice = anthropic.ToolChoiceUnionParam{}
	}
}
func (ai *Anthropic) SetMaxTokens(i int64)     { ai.maxTokens = &i }
func (ai *Anthropic) SetTemperature(f float64) { ai.temperature = &f }
func (ai *Anthropic) SetTopP(f float64)        { ai.topP = &f }

func (ai *Anthropic) SetCount(i int64) {
	fmt.Println("Anthropic doesn't support SetCount")
}
func (ai *Anthropic) SetJSONResponse(_ bool, _ *ai.JSONSchema) {
	fmt.Println("Anthropic currently doesn't support SetJSONResponse")
}

var _ ai.ChatResponse = new(ChatResponse[*anthropic.Message])

type ChatCompletionResponse interface {
	*anthropic.Message | anthropic.MessageStreamEventUnion
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
			if v, ok := i.AsAny().(anthropic.TextBlock); ok {
				res = append(res, v.Text)
			}
		}
	case anthropic.MessageStreamEventUnion:
		switch v := v.AsAny().(type) {
		case anthropic.ContentBlockDeltaEvent:
			if v, ok := v.Delta.AsAny().(anthropic.TextDelta); ok {
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
			if v, ok := i.AsAny().(anthropic.ToolUseBlock); ok {
				res = append(res, ai.FunctionCall{ID: v.ID, Name: v.Name, Arguments: string(v.Input)})
			}
		}
	case anthropic.MessageStreamEventUnion:
		switch v := v.AsAny().(type) {
		case anthropic.ContentBlockDeltaEvent:
			if v, ok := v.Delta.AsAny().(anthropic.InputJSONDelta); ok {
				if v.PartialJSON != "" {
					res = append(res, ai.FunctionCall{Arguments: v.PartialJSON})
				}
			}
		}
		if v, ok := v.ContentBlock.AsAny().(anthropic.ToolUseBlock); ok {
			res = append(res, ai.FunctionCall{ID: v.ID, Name: v.Name, Arguments: string(v.Input)})
		}
	}
	return
}

func (resp *ChatResponse[Response]) TokenCount() (res ai.TokenCount) {
	switch v := any(resp.resp).(type) {
	case *anthropic.Message:
		res.Prompt = v.Usage.InputTokens
		res.Result = v.Usage.OutputTokens
	case anthropic.MessageStreamEventUnion:
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

func toImageBlock(img ai.Image) anthropic.ContentBlockParamUnion {
	mime, data := img.Data()
	return anthropic.NewImageBlockBase64(mime, base64.StdEncoding.EncodeToString(data))
}

func fromImageBlockSource(img anthropic.ImageBlockParamSourceUnion) ai.Image {
	if src := img.OfBase64; src != nil {
		b, err := base64.StdEncoding.DecodeString(src.Data)
		if err != nil {
			panic(err)
		}
		return ai.ImageData(string(src.MediaType), b)
	} else if src := img.OfURL; src != nil {
		return ai.Image(src.URL)
	}
	panic(fmt.Sprintf("bad image: %v", img))
}

func (c *Anthropic) createRequest(
	history []anthropic.MessageParam,
	messages ...ai.Part,
) (req anthropic.MessageNewParams) {
	req.Model = c.model
	if !param.IsOmitted(c.toolChoice) {
		req.ToolChoice = c.toolChoice
	}
	if len(c.tools) > 0 {
		req.Tools = c.tools
	}
	if c.maxTokens != nil {
		req.MaxTokens = *c.maxTokens
	} else {
		req.MaxTokens = DefaultMaxTokens
	}
	if c.temperature != nil {
		req.Temperature = anthropic.Float(*c.temperature)
	}
	if c.topP != nil {
		req.TopP = anthropic.Float(*c.topP)
	}
	var msgs []anthropic.MessageParam
	for _, i := range history {
		var content []anthropic.ContentBlockParamUnion
		for _, v := range i.Content {
			if v.OfToolUse != nil {
				continue
			}
			content = append(content, v)
		}
		msgs = append(msgs, anthropic.MessageParam{
			Role:    i.Role,
			Content: content,
		})
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
	req.Messages = msgs
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
	stream  *ssestream.Stream[anthropic.MessageStreamEventUnion]
	session *ChatSession
	message anthropic.Message
}

func (cs *ChatStream) Next() (ai.ChatResponse, error) {
	if cs.stream.Next() {
		resp := cs.stream.Current()
		if cs.session != nil {
			err := cs.message.Accumulate(resp)
			if err != nil {
				return nil, err
			}
		}
		return &ChatResponse[anthropic.MessageStreamEventUnion]{resp}, nil
	}
	if err := cs.stream.Err(); err != nil {
		cs.message = anthropic.Message{}
		return nil, err
	}
	if cs.session != nil {
		cs.session.history = append(cs.session.history, cs.message.ToParam())
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
) (*ssestream.Stream[anthropic.MessageStreamEventUnion], error) {
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
	return &ChatStream{stream: stream}, nil
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
	return &ChatStream{stream: stream, session: session}, nil
}

func (session *ChatSession) History() (history []ai.Content) {
	for _, i := range session.history {
		for _, v := range i.Content {
			if v.OfText != nil {
				if text := v.OfText.Text; text != "" {
					history = append(history, ai.Content{Role: string(i.Role), Parts: []ai.Part{ai.Text(text)}})
				}
			}
			if v.OfImage != nil {
				history = append(history, ai.Content{Role: string(i.Role), Parts: []ai.Part{
					fromImageBlockSource(v.OfImage.Source),
				}})
			}
			if v.OfToolUse != nil {
				args, err := json.Marshal(v.OfToolUse.Input)
				if err != nil {
					panic(err)
				}
				history = append(history, ai.Content{Role: string(i.Role), Parts: []ai.Part{ai.FunctionCall{
					ID:        v.OfToolUse.ID,
					Name:      v.OfToolUse.Name,
					Arguments: string(args),
				}}})
			}
			if v.OfToolResult != nil {
				for _, ii := range v.OfToolResult.Content {
					if ii.OfText != nil {
						history = append(history, ai.Content{Role: string(i.Role), Parts: []ai.Part{ai.FunctionResponse{
							ID:       v.OfToolResult.ToolUseID,
							Response: ii.OfText.Text,
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
