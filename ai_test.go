package ai_test

import (
	"context"
	"fmt"
	"io"
	"os"
	"regexp"
	"testing"
	"time"

	"github.com/sunshineplan/ai"
	"github.com/sunshineplan/ai/chatgpt"
	"github.com/sunshineplan/ai/gemini"
)

func testChat(c ai.AI, prompt string) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println(prompt)
	resp, err := c.Chat(ctx, ai.Text(prompt))
	if err != nil {
		return err
	}
	fmt.Println(resp.Results())
	fmt.Println(resp.TokenCount())
	fmt.Println("---")
	return nil
}

func testChatStream(c ai.AI, prompt string) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println(prompt)
	stream, err := c.ChatStream(ctx, ai.Text(prompt))
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
		fmt.Println(resp.TokenCount())
	}
	fmt.Println("---")
	return nil
}

func testChatSession(c ai.AI) error {
	s := c.ChatSession()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("Hello, I have 2 dogs in my house.")
	resp, err := s.Chat(ctx, ai.Text("Hello, I have 2 dogs in my house."))
	if err != nil {
		return err
	}
	fmt.Println(resp.Results())
	ctx, cancel = context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("How many paws are in my house?")
	stream, err := s.ChatStream(ctx, ai.Text("How many paws are in my house?"))
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
		fmt.Println(resp.TokenCount())
	}
	fmt.Println("---")
	fmt.Println("History")
	for _, i := range s.History() {
		for _, p := range i.Parts {
			switch v := p.(type) {
			case ai.Text:
				fmt.Println(i.Role, ":", v)
			case ai.Image:
				fmt.Printf("%s : [%s]", i.Role, v.MIMEType)
			}
		}
	}
	fmt.Println("---")
	return nil
}

func checkMatch(t *testing.T, got string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		re, err := regexp.Compile("(?i:" + want + ")")
		if err != nil {
			t.Fatal(err)
		}
		if !re.MatchString(got) {
			t.Errorf("\ngot %q\nwanted to match %q", got, want)
		}
	}
}

func TestGemini(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return
	}
	gemini, err := gemini.New(
		context.Background(),
		ai.WithAPIKey(apiKey),
		ai.WithEndpoint(os.Getenv("GEMINI_ENDPOINT")),
		ai.WithProxy(os.Getenv("GEMINI_PROXY")),
		ai.WithModel(os.Getenv("GEMINI_MODEL")),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer gemini.Close()
	if err := testChat(gemini, "Hello!"); err != nil {
		t.Error(err)
	}
	if err := testChatStream(gemini, "Who am I?"); err != nil {
		t.Error(err)
	}
	if err := testChatSession(gemini); err != nil {
		t.Error(err)
	}
	img, err := os.ReadFile("testdata/personWorkingOnComputer.jpg")
	if err == nil {
		resp, err := gemini.Chat(context.Background(), ai.ImageData("image/jpeg", img), ai.Text("What is in this picture?"))
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(resp.Results())
		checkMatch(t, resp.Results()[0], "man|person", "computer|laptop")
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
		ai.WithModel(os.Getenv("CHATGPT_MODEL")),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer chatgpt.Close()
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
