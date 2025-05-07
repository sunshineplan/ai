package gemini

import (
	"context"
	"encoding/json"
	"io"
	"iter"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sunshineplan/ai"

	"golang.org/x/time/rate"
	"google.golang.org/genai"
)

const defaultModel = "gemini-2.0-flash"

var _ ai.AI = new(Gemini)

type Gemini struct {
	*genai.Client
	model  string
	config *genai.GenerateContentConfig

	limiter *rate.Limiter
}

func New(ctx context.Context, opts ...ai.ClientOption) (ai.AI, error) {
	cfg := new(ai.ClientConfig)
	for _, i := range opts {
		i.Apply(cfg)
	}
	cc := new(genai.ClientConfig)
	if cfg.Proxy == "" {
		cc.APIKey = cfg.APIKey
	} else {
		u, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, err
		}
		if t, ok := http.DefaultTransport.(*http.Transport); ok {
			t = t.Clone()
			t.Proxy = http.ProxyURL(u)
			cc.HTTPClient = &http.Client{Transport: t}
		}
	}
	if cfg.Endpoint != "" {
		cc.HTTPOptions.BaseURL = cfg.Endpoint
	}
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, err
	}
	c := NewWithClient(client, cfg.Model)
	if cfg.Limit != nil {
		c.SetLimit(*cfg.Limit)
	}
	ai.ApplyModelConfig(c, cfg.ModelConfig)
	return c, nil
}

func NewWithClient(client *genai.Client, model string) ai.AI {
	if model == "" {
		model = defaultModel
	}
	return &Gemini{Client: client, model: model, config: new(genai.GenerateContentConfig)}
}

func (Gemini) LLMs() ai.LLMs {
	return ai.Gemini
}

func (gemini *Gemini) Model() string {
	return gemini.model
}

func (gemini *Gemini) SetLimit(rpm int64) {
	gemini.limiter = ai.NewLimiter(rpm)
}

func (gemini *Gemini) Limit() (rpm int64) {
	if gemini.limiter == nil {
		return math.MaxInt64
	}
	return int64(gemini.limiter.Limit() / rate.Every(time.Minute))
}

func (ai *Gemini) wait(ctx context.Context) error {
	if ai.limiter != nil {
		return ai.limiter.Wait(ctx)
	}
	return nil
}

func (ai *Gemini) SetModel(model string) {
	ai.model = model
}

func genaiSchema(schema *ai.Schema) (*genai.Schema, error) {
	if schema == nil {
		return nil, nil
	}
	p, err := genaiProperties(schema.Properties)
	if err != nil {
		return nil, err
	}
	var items *genai.Schema
	if schema.Items != nil {
		p, err := genaiProperties(schema.Items.Properties)
		if err != nil {
			return nil, err
		}
		items = &genai.Schema{
			Type:       genaiType(schema.Items.Type),
			Properties: p,
			Enum:       schema.Items.Enum,
			Required:   schema.Items.Required,
		}
	}
	return &genai.Schema{
		Type:       genaiType(schema.Type),
		Properties: p,
		Enum:       schema.Enum,
		Items:      items,
		Required:   schema.Required,
	}, nil
}

func (gemini *Gemini) SetFunctionCall(f []ai.Function, mode ai.FunctionCallingMode) {
	if len(f) == 0 {
		gemini.config.Tools = nil
		gemini.config.ToolConfig = nil
		return
	}
	var declarations []*genai.FunctionDeclaration
	for _, i := range f {
		schema, err := genaiSchema(&i.Parameters)
		if err != nil {
			continue
		}
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name:        i.Name,
			Description: i.Description,
			Parameters:  schema,
		})
	}
	gemini.config.Tools = []*genai.Tool{{FunctionDeclarations: declarations}}
	switch mode {
	case ai.FunctionCallingAuto:
		gemini.config.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAuto},
		}
	case ai.FunctionCallingAny:
		gemini.config.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny},
		}
	case ai.FunctionCallingNone:
		gemini.config.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone},
		}
	default:
		gemini.config.ToolConfig = nil
	}
}
func (ai *Gemini) SetCount(i int64)         { ai.config.CandidateCount = int32(i) }
func (ai *Gemini) SetMaxTokens(i int64)     { ai.config.MaxOutputTokens = int32(i) }
func (ai *Gemini) SetTemperature(f float64) { ai.config.Temperature = genai.Ptr(float32(f)) }
func (ai *Gemini) SetTopP(f float64)        { ai.config.TopP = genai.Ptr(float32(f)) }
func (ai *Gemini) SetJSONResponse(set bool, schema *ai.JSONSchema) {
	if set {
		ai.config.ResponseMIMEType = "application/json"
		if schema != nil {
			ai.config.ResponseSchema, _ = genaiSchema(&schema.Schema)
		} else {
			ai.config.ResponseSchema = nil
		}
	} else {
		ai.config.ResponseMIMEType = "text/plain"
		ai.config.ResponseSchema = nil
	}
}

func toParts(src []ai.Part) (dst []*genai.Part) {
	for _, i := range src {
		switch v := i.(type) {
		case ai.Text:
			dst = append(dst, genai.NewPartFromText(string(v)))
		case ai.Image:
			mime, data := v.Data()
			dst = append(dst, genai.NewPartFromBytes(data, mime))
		case ai.Blob:
			dst = append(dst, genai.NewPartFromBytes(v.Data, v.MIMEType))
		case ai.FunctionCall:
			b, err := json.Marshal(v.Arguments)
			if err != nil {
				panic(err)
			}
			var args map[string]any
			if err := json.Unmarshal(b, &args); err != nil {
				panic(err)
			}
			dst = append(dst, genai.NewPartFromFunctionCall(v.Name, args))
		case ai.FunctionResponse:
			var resp map[string]any
			if err := json.Unmarshal([]byte(v.Response), &resp); err != nil {
				panic(err)
			}
			dst = append(dst, genai.NewPartFromFunctionResponse(v.ID, resp))
		}
	}
	return
}

func fromParts(src []*genai.Part) (dst []ai.Part) {
	for _, i := range src {
		if i.Text != "" {
			dst = append(dst, ai.Text(i.Text))
		} else if i.InlineData != nil {
			dst = append(dst, ai.Blob{MIMEType: i.InlineData.MIMEType, Data: i.InlineData.Data})
		} else if i.FunctionCall != nil {
			b, err := json.Marshal(i.FunctionCall.Args)
			if err != nil {
				panic(err)
			}
			id := i.FunctionCall.ID
			if id == "" {
				id = i.FunctionCall.Name
			}
			dst = append(dst, ai.FunctionCall{ID: id, Name: i.FunctionCall.Name, Arguments: string(b)})
		} else if i.FunctionResponse != nil {
			b, err := json.Marshal(i.FunctionResponse.Response)
			if err != nil {
				panic(err)
			}
			id := i.FunctionResponse.ID
			if id == "" {
				id = i.FunctionResponse.Name
			}
			dst = append(dst, ai.FunctionResponse{ID: id, Response: string(b)})
		}
	}
	return
}

var _ ai.ChatResponse = new(ChatResponse)

type ChatResponse struct {
	*genai.GenerateContentResponse
}

func (resp *ChatResponse) Raw() any {
	return resp.GenerateContentResponse
}

func (resp *ChatResponse) Results() (res []string) {
	for _, i := range resp.Candidates {
		if i.Content != nil {
			var s []string
			for _, i := range i.Content.Parts {
				if i.Text != "" {
					s = append(s, i.Text)
				}
			}
			res = append(res, strings.Join(s, "\n"))
		}
	}
	return
}

func (resp *ChatResponse) FunctionCalls() (res []ai.FunctionCall) {
	for _, i := range resp.Candidates {
		if i.Content != nil {
			for _, i := range i.Content.Parts {
				if i.FunctionCall != nil {
					b, err := json.Marshal(i.FunctionCall.Args)
					if err != nil {
						panic(err)
					}
					id := i.FunctionCall.ID
					if id == "" {
						id = i.FunctionCall.Name
					}
					res = append(res, ai.FunctionCall{ID: id, Name: i.FunctionCall.Name, Arguments: string(b)})
				}
			}
		}
	}
	return
}

func (resp *ChatResponse) TokenCount() (res ai.TokenCount) {
	if usage := resp.UsageMetadata; usage != nil {
		res.Prompt = int64(usage.PromptTokenCount)
		res.Result = int64(usage.CandidatesTokenCount)
		res.Total = int64(usage.TotalTokenCount)
	}
	return
}

func (resp *ChatResponse) String() string {
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

func (ai *Gemini) Chat(ctx context.Context, parts ...ai.Part) (ai.ChatResponse, error) {
	if err := ai.wait(ctx); err != nil {
		return nil, err
	}
	resp, err := ai.Models.GenerateContent(
		ctx,
		ai.model,
		[]*genai.Content{genai.NewContentFromParts(toParts(parts), genai.RoleUser)},
		ai.config,
	)
	if err != nil {
		return nil, err
	}
	return &ChatResponse{resp}, nil
}

var _ ai.ChatStream = new(ChatStream)

type ChatStream struct {
	next func() (*genai.GenerateContentResponse, error, bool)
	stop func()
}

func (stream *ChatStream) Next() (ai.ChatResponse, error) {
	resp, err, ok := stream.next()
	if !ok {
		return nil, io.EOF
	}
	if err != nil {
		return nil, err
	}
	return &ChatResponse{resp}, nil
}

func (stream *ChatStream) Close() error {
	stream.stop()
	return nil
}

func (ai *Gemini) ChatStream(ctx context.Context, parts ...ai.Part) (ai.ChatStream, error) {
	if err := ai.wait(ctx); err != nil {
		return nil, err
	}
	next, stop := iter.Pull2(ai.Models.GenerateContentStream(
		ctx,
		ai.model,
		[]*genai.Content{genai.NewContentFromParts(toParts(parts), genai.RoleUser)},
		ai.config,
	))
	return &ChatStream{next, stop}, nil
}

var _ ai.ChatSession = new(ChatSession)

type ChatSession struct {
	ai *Gemini
	cs *genai.Chat
}

func (session *ChatSession) Chat(ctx context.Context, parts ...ai.Part) (ai.ChatResponse, error) {
	if err := session.ai.wait(ctx); err != nil {
		return nil, err
	}
	var genaiParts []genai.Part
	for _, i := range toParts(parts) {
		genaiParts = append(genaiParts, *i)
	}
	resp, err := session.cs.SendMessage(ctx, genaiParts...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse{resp}, nil
}

func (session *ChatSession) ChatStream(ctx context.Context, parts ...ai.Part) (ai.ChatStream, error) {
	if err := session.ai.wait(ctx); err != nil {
		return nil, err
	}
	var genaiParts []genai.Part
	for _, i := range toParts(parts) {
		genaiParts = append(genaiParts, *i)
	}
	next, stop := iter.Pull2(session.cs.SendMessageStream(ctx, genaiParts...))
	return &ChatStream{next, stop}, nil
}

func (session *ChatSession) History() (history []ai.Content) {
	for _, i := range session.cs.History(false) {
		history = append(history, ai.Content{Parts: fromParts(i.Parts), Role: i.Role})
	}
	return
}

func (ai *Gemini) ChatSession() ai.ChatSession {
	chat, _ := ai.Chats.Create(context.Background(), ai.model, ai.config, nil)
	return &ChatSession{ai, chat}
}

func (ai *Gemini) Close() error {
	ai.Client = nil
	return nil
}
