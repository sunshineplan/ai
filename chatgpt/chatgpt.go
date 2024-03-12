package chatgpt

import (
	"context"
	"io"

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

	limiter *rate.Limiter
}

func New(authToken string) ai.AI {
	return NewWithClient(openai.NewClient(authToken))
}

func NewWithClient(client *openai.Client) ai.AI {
	return &ChatGPT{Client: client, model: defaultModel}
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

func (ai *ChatGPT) createRequest(history []openai.ChatCompletionMessage, messages ...string) (req openai.ChatCompletionRequest) {
	req.Model = ai.model
	if ai.maxTokens != nil {
		req.MaxTokens = int(*ai.maxTokens)
	}
	if ai.count != nil {
		req.N = int(*ai.count)
	}
	if ai.temperature != nil {
		req.Temperature = *ai.temperature
	}
	if ai.topP != nil {
		req.TopP = *ai.topP
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

func (ai *ChatGPT) chat(
	ctx context.Context,
	history []openai.ChatCompletionMessage,
	messages ...string,
) (resp openai.ChatCompletionResponse, err error) {
	if err = ai.wait(ctx); err != nil {
		return
	}
	return ai.CreateChatCompletion(ctx, ai.createRequest(history, messages...))
}

func (ai *ChatGPT) Chat(ctx context.Context, messages ...string) (ai.ChatResponse, error) {
	resp, err := ai.chat(ctx, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatResponse[openai.ChatCompletionResponse]{resp}, nil
}

var _ ai.ChatStream = new(ChatStream)

type ChatStream struct {
	*openai.ChatCompletionStream
	cs      *ChatSession
	content string
}

func (stream *ChatStream) Next() (ai.ChatResponse, error) {
	resp, err := stream.Recv()
	if err != nil {
		if err == io.EOF {
			if stream.cs != nil {
				stream.cs.History = append(stream.cs.History, openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleAssistant, Content: stream.content})
			}
		}
		stream.content = ""
		return nil, err
	}
	if stream.cs != nil {
		stream.content += resp.Choices[0].Delta.Content
	}
	return &ChatResponse[openai.ChatCompletionStreamResponse]{resp}, nil
}

func (ai *ChatGPT) chatStream(
	ctx context.Context,
	history []openai.ChatCompletionMessage,
	messages ...string,
) (*openai.ChatCompletionStream, error) {
	if err := ai.wait(ctx); err != nil {
		return nil, err
	}
	req := ai.createRequest(history, messages...)
	req.Stream = true
	return ai.CreateChatCompletionStream(ctx, req)
}

func (ai *ChatGPT) ChatStream(ctx context.Context, messages ...string) (ai.ChatStream, error) {
	stream, err := ai.chatStream(ctx, nil, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatStream{stream, nil, ""}, nil
}

var _ ai.Chatbot = new(ChatSession)

type ChatSession struct {
	ai      *ChatGPT
	History []openai.ChatCompletionMessage
}

func (session *ChatSession) Chat(ctx context.Context, messages ...string) (ai.ChatResponse, error) {
	resp, err := session.ai.chat(ctx, session.History, messages...)
	if err != nil {
		return nil, err
	}
	session.History = append(session.History, resp.Choices[0].Message)
	return &ChatResponse[openai.ChatCompletionResponse]{resp}, nil
}

func (session *ChatSession) ChatStream(ctx context.Context, messages ...string) (ai.ChatStream, error) {
	stream, err := session.ai.chatStream(ctx, session.History, messages...)
	if err != nil {
		return nil, err
	}
	return &ChatStream{stream, session, ""}, nil
}

func (ai *ChatGPT) ChatSession() ai.Chatbot {
	ai.count = nil
	return &ChatSession{ai: ai}
}

func (ai *ChatGPT) Close() error {
	return nil
}
