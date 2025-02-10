package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sunshineplan/ai"

	"github.com/google/generative-ai-go/genai"
	"golang.org/x/time/rate"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

const defaultModel = "gemini-2.0-flash"

var _ ai.AI = new(Gemini)

type Gemini struct {
	*genai.Client
	model  *genai.GenerativeModel
	config genai.GenerationConfig

	limiter *rate.Limiter
}

func New(ctx context.Context, opts ...ai.ClientOption) (ai.AI, error) {
	cfg := new(ai.ClientConfig)
	for _, i := range opts {
		i.Apply(cfg)
	}
	var o []option.ClientOption
	if cfg.Proxy == "" {
		o = append(o, option.WithAPIKey(cfg.APIKey))
	} else {
		u, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, err
		}
		if t, ok := http.DefaultTransport.(*http.Transport); ok {
			t = t.Clone()
			t.Proxy = http.ProxyURL(u)
			o = append(o, option.WithHTTPClient(&http.Client{Transport: &apikey{cfg.APIKey, t}}))
		}
	}
	if cfg.Endpoint != "" {
		o = append(o, option.WithEndpoint(cfg.Endpoint))
	}
	client, err := genai.NewClient(ctx, o...)
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
	return &Gemini{Client: client, model: client.GenerativeModel(model)}
}

func (Gemini) LLMs() ai.LLMs {
	return ai.Gemini
}

func (gemini *Gemini) Model(ctx context.Context) (string, error) {
	info, err := gemini.model.Info(ctx)
	if err != nil {
		return "", err
	}
	return info.Name, nil
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
	ai.model = ai.GenerativeModel(model)
	ai.model.GenerationConfig = ai.config
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
		gemini.model.Tools = nil
		gemini.model.ToolConfig = nil
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
	gemini.model.Tools = []*genai.Tool{{FunctionDeclarations: declarations}}
	switch mode {
	case ai.FunctionCallingAuto:
		gemini.model.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingAuto},
		}
	case ai.FunctionCallingAny:
		gemini.model.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingAny},
		}
	case ai.FunctionCallingNone:
		gemini.model.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingNone},
		}
	default:
		gemini.model.ToolConfig = nil
	}
}
func (ai *Gemini) SetCount(i int64) {
	ai.config.SetCandidateCount(int32(i))
	ai.model.GenerationConfig = ai.config
}
func (ai *Gemini) SetMaxTokens(i int64) {
	ai.config.SetMaxOutputTokens(int32(i))
	ai.model.GenerationConfig = ai.config
}
func (ai *Gemini) SetTemperature(f float64) {
	ai.config.SetTemperature(float32(f))
	ai.model.GenerationConfig = ai.config
}
func (ai *Gemini) SetTopP(f float64) {
	ai.config.SetTopP(float32(f))
	ai.model.GenerationConfig = ai.config
}
func (ai *Gemini) SetJSONResponse(set bool, schema *ai.JSONSchema) {
	if set {
		ai.config.ResponseMIMEType = "application/json"
		if schema != nil {
			ai.config.ResponseSchema, _ = genaiSchema(&schema.Schema)
		}
	} else {
		ai.config.ResponseMIMEType = "text/plain"
	}
	ai.model.GenerationConfig = ai.config
}

func toParts(src []ai.Part) (dst []genai.Part) {
	for _, i := range src {
		switch v := i.(type) {
		case ai.Text:
			dst = append(dst, genai.Text(v))
		case ai.Image:
			mime, data := v.Data()
			dst = append(dst, genai.Blob{MIMEType: mime, Data: data})
		case ai.Blob:
			dst = append(dst, genai.Blob(v))
		case ai.FunctionCall:
			b, err := json.Marshal(v.Arguments)
			if err != nil {
				panic(err)
			}
			var args map[string]any
			if err := json.Unmarshal(b, &args); err != nil {
				panic(err)
			}
			dst = append(dst, genai.FunctionCall{Name: v.ID, Args: args})
		case ai.FunctionResponse:
			var resp map[string]any
			if err := json.Unmarshal([]byte(v.Response), &resp); err != nil {
				panic(err)
			}
			dst = append(dst, genai.FunctionResponse{Name: v.ID, Response: resp})
		}
	}
	return
}

func fromParts(src []genai.Part) (dst []ai.Part) {
	for _, i := range src {
		switch v := i.(type) {
		case genai.Text:
			dst = append(dst, ai.Text(v))
		case genai.Blob:
			dst = append(dst, ai.Blob(v))
		case genai.FunctionCall:
			b, err := json.Marshal(v.Args)
			if err != nil {
				panic(err)
			}
			dst = append(dst, ai.FunctionCall{ID: v.Name, Name: v.Name, Arguments: string(b)})
		case genai.FunctionResponse:
			b, err := json.Marshal(v.Response)
			if err != nil {
				panic(err)
			}
			dst = append(dst, ai.FunctionResponse{ID: v.Name, Response: string(b)})
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
				if v, ok := i.(genai.Text); ok {
					s = append(s, string(v))
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
				if v, ok := i.(genai.FunctionCall); ok {
					b, err := json.Marshal(v.Args)
					if err != nil {
						panic(err)
					}
					res = append(res, ai.FunctionCall{ID: v.Name, Name: v.Name, Arguments: string(b)})
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
	resp, err := ai.model.GenerateContent(ctx, toParts(parts)...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse{resp}, nil
}

var _ ai.ChatStream = new(ChatStream)

type ChatStream struct {
	iter *genai.GenerateContentResponseIterator
}

func (stream *ChatStream) Next() (ai.ChatResponse, error) {
	if stream.iter == nil {
		return nil, errors.New("stream iterator is nil or already closed")
	}
	resp, err := stream.iter.Next()
	if err != nil {
		if err == iterator.Done {
			return nil, io.EOF
		}
		return nil, err
	}
	return &ChatResponse{resp}, nil
}

func (stream *ChatStream) Close() error {
	stream.iter = nil
	return nil
}

func (ai *Gemini) ChatStream(ctx context.Context, parts ...ai.Part) (ai.ChatStream, error) {
	if err := ai.wait(ctx); err != nil {
		return nil, err
	}
	return &ChatStream{ai.model.GenerateContentStream(ctx, toParts(parts)...)}, nil
}

var _ ai.ChatSession = new(ChatSession)

type ChatSession struct {
	ai *Gemini
	cs *genai.ChatSession
}

func (session *ChatSession) Chat(ctx context.Context, parts ...ai.Part) (ai.ChatResponse, error) {
	if err := session.ai.wait(ctx); err != nil {
		return nil, err
	}
	resp, err := session.cs.SendMessage(ctx, toParts(parts)...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse{resp}, nil
}

func (session *ChatSession) ChatStream(ctx context.Context, parts ...ai.Part) (ai.ChatStream, error) {
	if err := session.ai.wait(ctx); err != nil {
		return nil, err
	}
	return &ChatStream{session.cs.SendMessageStream(ctx, toParts(parts)...)}, nil
}

func (session *ChatSession) History() (history []ai.Content) {
	for _, i := range session.cs.History {
		history = append(history, ai.Content{Parts: fromParts(i.Parts), Role: i.Role})
	}
	return
}

func (ai *Gemini) ChatSession() ai.ChatSession {
	return &ChatSession{ai, ai.model.StartChat()}
}
