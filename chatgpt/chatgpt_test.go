package chatgpt

import (
	"context"
	"fmt"
	"io"
	"os"
	"testing"
	"time"
)

func TestChatGPT(t *testing.T) {
	apiKey := os.Getenv("CHATGPT_API_KEY")
	if apiKey == "" {
		return
	}
	chatgpt := New(apiKey)
	defer chatgpt.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("Who are you?")
	resp, err := chatgpt.Chat(ctx, "Who are you?")
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(resp.Results())
	fmt.Println("---")
	fmt.Println("Who am I?")
	ctx, cancel = context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	stream, err := chatgpt.ChatStream(ctx, "Who am I?")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
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
	s := chatgpt.ChatSession()
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
	defer stream.Close()
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
	for _, i := range s.History() {
		fmt.Println(i.Role, ":", i.Content)
	}
}
