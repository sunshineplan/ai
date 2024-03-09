package gemini

import (
	"ai"
	"context"
	"fmt"
	"io"
	"os"
	"testing"
	"time"
)

func TestGemini(t *testing.T) {
	if proxy := os.Getenv("GEMINI_PROXY"); proxy != "" {
		ai.SetProxy(proxy)
	}
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return
	}
	gemini, err := New(apiKey)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("Who are you?")
	resp, err := gemini.Chat(ctx, "Who are you?")
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(resp.Results())
	fmt.Println("---")
	fmt.Println("Who am I?")
	ctx, cancel = context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	stream, err := gemini.ChatStream(ctx, "Who am I?")
	if err != nil {
		t.Fatal(err)
	}
	for {
		resp, err := stream.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		fmt.Println(resp.Results())
	}
	fmt.Println("---")
	s := gemini.ChatSession()
	ctx, cancel = context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("Hello, I have 2 dogs in my house.")
	resp, err = s.Chat(ctx, "Hello, I have 2 dogs in my house.")
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(resp.Results())
	ctx, cancel = context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("How many paws are in my house?")
	stream, err = s.ChatStream(ctx, "How many paws are in my house?")
	if err != nil {
		t.Fatal(err)
	}
	for {
		resp, err := stream.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		fmt.Println(resp.Results())
	}
}
