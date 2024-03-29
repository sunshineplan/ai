package ai_test

import (
	"context"
	"fmt"
	"io"
	"os"
	"testing"
	"time"

	"github.com/sunshineplan/ai"
	"github.com/sunshineplan/ai/chatgpt"
	"github.com/sunshineplan/ai/gemini"
)

func testChat(ai ai.AI, prompt string) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println(prompt)
	resp, err := ai.Chat(ctx, prompt)
	if err != nil {
		return err
	}
	fmt.Println(resp.Results())
	fmt.Println("---")
	return nil
}

func testChatStream(ai ai.AI, prompt string) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println(prompt)
	stream, err := ai.ChatStream(ctx, prompt)
	if err != nil {
		return err
	}
	defer stream.Close()
	for {
		resp, err := stream.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		fmt.Println(resp.Results())
	}
	fmt.Println("---")
	return nil
}

func testChatSession(ai ai.AI) error {
	s := ai.ChatSession()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("Hello, I have 2 dogs in my house.")
	resp, err := s.Chat(ctx, "Hello, I have 2 dogs in my house.")
	if err != nil {
		return err
	}
	fmt.Println(resp.Results())
	ctx, cancel = context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("How many paws are in my house?")
	stream, err := s.ChatStream(ctx, "How many paws are in my house?")
	if err != nil {
		return err
	}
	defer stream.Close()
	for {
		resp, err := stream.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		fmt.Println(resp.Results())
	}
	fmt.Println("---")
	fmt.Println("History")
	for _, i := range s.History() {
		fmt.Println(i.Role, ":", i.Content)
	}
	fmt.Println("---")
	return nil
}

func TestGemini(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return
	}
	gemini, err := gemini.New(
		ai.WithAPIKey(apiKey),
		ai.WithEndpoint(os.Getenv("GEMINI_ENDPOINT")),
		ai.WithProxy(os.Getenv("GEMINI_PROXY")),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer gemini.Close()
	if model := os.Getenv("GEMINI_MODEL"); model != "" {
		gemini.SetModel(model)
	}
	if err := testChat(gemini, "Who are you?"); err != nil {
		t.Error(err)
	}
	if err := testChatStream(gemini, "Who am I?"); err != nil {
		t.Error(err)
	}
	if err := testChatSession(gemini); err != nil {
		t.Error(err)
	}
}

func TestChatGPT(t *testing.T) {
	apiKey := os.Getenv("CHATGPT_API_KEY")
	if apiKey == "" {
		return
	}
	chatgpt, err := chatgpt.New(
		ai.WithAPIKey(apiKey),
		ai.WithEndpoint(os.Getenv("CHATGPT_ENDPOINT")),
		ai.WithProxy(os.Getenv("CHATGPT_PROXY")),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer chatgpt.Close()
	if model := os.Getenv("CHATGPT_MODEL"); model != "" {
		chatgpt.SetModel(model)
	}
	if err := testChat(chatgpt, "Who are you?"); err != nil {
		t.Error(err)
	}
	if err := testChatStream(chatgpt, "Who am I?"); err != nil {
		t.Error(err)
	}
	if err := testChatSession(chatgpt); err != nil {
		t.Error(err)
	}
}
