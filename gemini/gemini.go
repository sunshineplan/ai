package gemini

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/sunshineplan/ai"

	"github.com/google/generative-ai-go/genai"
	"golang.org/x/time/rate"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

const defaultModel = "gemini-1.0-pro"

var _ ai.AI = new(Gemini)

type Gemini struct {
	c      *genai.Client
	model  *genai.GenerativeModel
	config genai.GenerationConfig

	limiter *rate.Limiter
}

func New(opts ...ai.ClientOption) (ai.AI, error) {
	cfg := new(ai.ClientConfig)
	for _, i := range opts {
		i.Apply(cfg)
	}
	o := []option.ClientOption{option.WithAPIKey(cfg.APIKey)}
	if cfg.Proxy != "" {
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
	client, err := genai.NewClient(context.Background(), o...)
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
	return &Gemini{c: client, model: client.GenerativeModel(model)}
}

func (Gemini) LLMs() ai.LLMs {
	return ai.Gemini
}

func (gemini *Gemini) SetLimit(limit rate.Limit) {
	gemini.limiter = ai.NewLimiter(limit)
}

func (ai *Gemini) wait(ctx context.Context) error {
	if ai.limiter != nil {
		return ai.limiter.Wait(ctx)
	}
	return nil
}

func (ai *Gemini) SetModel(model string) {
	ai.model = ai.c.GenerativeModel(model)
	ai.model.GenerationConfig = ai.config
}

func (ai *Gemini) SetCount(i int32)         { ai.config.SetCandidateCount(i) }
func (ai *Gemini) SetMaxTokens(i int32)     { ai.config.SetMaxOutputTokens(i) }
func (ai *Gemini) SetTemperature(f float32) { ai.config.SetTemperature(f) }
func (ai *Gemini) SetTopP(f float32)        { ai.config.SetTopP(f) }

func texts2parts(texts []string) (parts []genai.Part) {
	for _, i := range texts {
		parts = append(parts, genai.Text(i))
	}
	return
}

func parts2texts(parts []genai.Part) (texts []string) {
	for _, i := range parts {
		if text, ok := i.(genai.Text); ok {
			texts = append(texts, string(text))
		}
	}
	return
}

var _ ai.ChatResponse = new(ChatResponse)

type ChatResponse struct {
	*genai.GenerateContentResponse
}

func (resp *ChatResponse) Results() (res []string) {
	for _, i := range resp.Candidates {
		if i.Content != nil {
			res = append(res, strings.Join(parts2texts(i.Content.Parts), "\n"))
		}
	}
	return
}

func (resp *ChatResponse) String() string {
	if res := resp.Results(); len(res) > 0 {
		return res[0]
	}
	return ""
}

func (ai *Gemini) Chat(ctx context.Context, parts ...string) (ai.ChatResponse, error) {
	if err := ai.wait(ctx); err != nil {
		return nil, err
	}
	resp, err := ai.model.GenerateContent(ctx, texts2parts(parts)...)
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

func (ai *Gemini) ChatStream(ctx context.Context, parts ...string) (ai.ChatStream, error) {
	if err := ai.wait(ctx); err != nil {
		return nil, err
	}
	return &ChatStream{ai.model.GenerateContentStream(ctx, texts2parts(parts)...)}, nil
}

var _ ai.ChatSession = new(ChatSession)

type ChatSession struct {
	ai *Gemini
	cs *genai.ChatSession
}

func (session *ChatSession) Chat(ctx context.Context, parts ...string) (ai.ChatResponse, error) {
	if err := session.ai.wait(ctx); err != nil {
		return nil, err
	}
	resp, err := session.cs.SendMessage(ctx, texts2parts(parts)...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse{resp}, nil
}

func (session *ChatSession) ChatStream(ctx context.Context, parts ...string) (ai.ChatStream, error) {
	if err := session.ai.wait(ctx); err != nil {
		return nil, err
	}
	return &ChatStream{session.cs.SendMessageStream(ctx, texts2parts(parts)...)}, nil
}

func (session *ChatSession) History() (history []ai.Message) {
	for _, i := range session.cs.History {
		history = append(history, ai.Message{Content: strings.Join(parts2texts(i.Parts), "\n"), Role: i.Role})
	}
	return
}

func (ai *Gemini) ChatSession() ai.ChatSession {
	return &ChatSession{ai, ai.model.StartChat()}
}

func (ai *Gemini) Close() error {
	return ai.c.Close()
}
