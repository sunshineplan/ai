package ai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/sunshineplan/ai"
	"github.com/sunshineplan/ai/anthropic"
	"github.com/sunshineplan/ai/chatgpt"
	"github.com/sunshineplan/ai/gemini"
)

func testChat(model string, c ai.AI, prompt string) error {
	if model == "" {
		return nil
	} else {
		c.SetModel(model)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println(prompt)
	resp, err := c.Chat(ctx, ai.Text(prompt))
	if err != nil {
		return err
	}
	fmt.Println(resp)
	fmt.Println(resp.TokenCount())
	fmt.Println("---")
	return nil
}

func testChatStream(model string, c ai.AI, prompt string) error {
	if model == "" {
		return nil
	} else {
		c.SetModel(model)
	}
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
		fmt.Println(resp)
		fmt.Println(resp.TokenCount())
	}
	fmt.Println("---")
	return nil
}

func testChatSession(model string, c ai.AI) error {
	if model == "" {
		return nil
	} else {
		c.SetModel(model)
	}
	s := c.ChatSession()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	fmt.Println("Hello, I have 2 dogs in my house.")
	resp, err := s.Chat(ctx, ai.Text("Hello, I have 2 dogs in my house."))
	if err != nil {
		return err
	}
	fmt.Println(resp)
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
		fmt.Println(resp)
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

func testFunctionCall(t *testing.T, model string, c ai.AI) {
	if model == "" {
		return
	} else {
		c.SetModel(model)
	}
	c.SetTemperature(0)
	movieChat := func(t *testing.T, s ai.Schema, fcm ai.FunctionCallingMode) {
		movieTool := ai.Function{
			Name:        "find_theaters",
			Description: "find theaters based on location and optionally movie title which is currently playing in theaters",
			Parameters:  s,
		}
		c.SetFunctionCall([]ai.Function{movieTool}, fcm)
		session := c.ChatSession()
		stream, err := session.ChatStream(context.Background(), ai.Text("Which theaters in Mountain View show Barbie movie?"))
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
			fmt.Println(resp)
			fmt.Println(resp.TokenCount())
		}
		var funcalls []ai.FunctionCall
		for _, i := range session.History() {
			for _, i := range i.Parts {
				if v, ok := i.(ai.FunctionCall); ok {
					funcalls = append(funcalls, v)
				}
			}
		}
		if fcm == ai.FunctionCallingNone {
			if len(funcalls) != 0 {
				t.Fatalf("got %d FunctionCalls, want 0", len(funcalls))
			}
			return
		}
		if len(funcalls) != 1 {
			t.Fatalf("got %d FunctionCalls, want 1", len(funcalls))
		}
		funcall := funcalls[0]
		if g, w := funcall.Name, movieTool.Name; g != w {
			t.Fatalf("FunctionCall.Name: got %q, want %q", g, w)
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(funcall.Arguments), &m); err != nil {
			t.Fatal(err)
		}
		locArg, ok := m["location"].(string)
		if !ok {
			t.Fatalf(`funcall.Arguments["location"] is not a string`)
		}
		if c := "Mountain View"; !strings.Contains(locArg, c) {
			t.Fatalf(`FunctionCall.Args["location"]: got %q, want string containing %q`, locArg, c)
		}
		stream, err = session.ChatStream(context.Background(), ai.Text("response:"), ai.FunctionResponse{
			ID:       funcall.ID,
			Response: `{"theater":"AMC16"}`,
		})
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
			fmt.Println(resp)
			fmt.Println(resp.TokenCount())
		}
		var res []string
		for _, i := range session.History() {
			for _, ii := range i.Parts {
				switch v := ii.(type) {
				case ai.Text:
					fmt.Println(i.Role, ":", v)
					res = append(res, string(v))
				case ai.FunctionResponse:
					fmt.Println(i.Role, ":", v.Response)
				}
			}
		}
		checkMatch(t, strings.Join(res, "/n"), "AMC")
	}
	schema := ai.Schema{
		Type: "object",
		Properties: map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616",
			},
			"title": map[string]any{
				"type":        "string",
				"description": "Any movie title",
			},
		},
		Required: []string{"location"},
	}
	t.Run("direct", func(t *testing.T) {
		movieChat(t, schema, ai.FunctionCallingAuto)
	})
	t.Run("none", func(t *testing.T) {
		movieChat(t, schema, ai.FunctionCallingNone)
	})
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
	)
	if err != nil {
		t.Fatal(err)
	}
	defer gemini.Close()
	model := os.Getenv("GEMINI_MODEL")
	if err := testChat(model, gemini, "Hello!"); err != nil {
		t.Error(err)
	}
	if err := testChatStream(model, gemini, "Who am I?"); err != nil {
		t.Error(err)
	}
	if err := testChatSession(model, gemini); err != nil {
		t.Error(err)
	}
	if modelForImage := os.Getenv("GEMINI_MODEL_FOR_IMAGE"); modelForImage != "" {
		gemini.SetModel(modelForImage)
	} else {
		return
	}
	img, err := os.ReadFile("testdata/personWorkingOnComputer.jpg")
	if err == nil {
		resp, err := gemini.Chat(context.Background(), ai.ImageData("image/jpeg", img), ai.Text("What is in this picture?"))
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(resp)
		checkMatch(t, resp.Results()[0], "man|person", "computer|laptop")
	}
	testFunctionCall(t, os.Getenv("GEMINI_MODEL_FOR_TOOLS"), gemini)
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
	model := os.Getenv("CHATGPT_MODEL")
	if err := testChat(model, chatgpt, "Who are you?"); err != nil {
		t.Error(err)
	}
	if err := testChatStream(model, chatgpt, "Who am I?"); err != nil {
		t.Error(err)
	}
	if err := testChatSession(model, chatgpt); err != nil {
		t.Error(err)
	}
	if modelForImage := os.Getenv("CHATGPT_MODEL_FOR_IMAGE"); modelForImage != "" {
		chatgpt.SetModel(modelForImage)
	} else {
		return
	}
	img, err := os.ReadFile("testdata/personWorkingOnComputer.jpg")
	if err == nil {
		resp, err := chatgpt.Chat(context.Background(), ai.ImageData("image/jpeg", img), ai.Text("What is in this picture?"))
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(resp)
		checkMatch(t, resp.Results()[0], "man|person", "computer|laptop")
	}
	testFunctionCall(t, os.Getenv("CHATGPT_MODEL_FOR_TOOLS"), chatgpt)
}

func TestAnthropic(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return
	}
	anthropic, err := anthropic.New(
		ai.WithAPIKey(apiKey),
		ai.WithEndpoint(os.Getenv("ANTHROPIC_ENDPOINT")),
		ai.WithProxy(os.Getenv("ANTHROPIC_PROXY")),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer anthropic.Close()
	model := os.Getenv("ANTHROPIC_MODEL")
	if err := testChat(model, anthropic, "Who are you?"); err != nil {
		t.Fatal(err)
	}
	if err := testChatStream(model, anthropic, "Who am I?"); err != nil {
		t.Error(err)
	}
	if err := testChatSession(model, anthropic); err != nil {
		t.Error(err)
	}
	if modelForImage := os.Getenv("ANTHROPIC_MODEL_FOR_IMAGE"); modelForImage != "" {
		anthropic.SetModel(modelForImage)
	} else {
		return
	}
	img, err := os.ReadFile("testdata/personWorkingOnComputer.jpg")
	if err == nil {
		resp, err := anthropic.Chat(context.Background(), ai.ImageData("image/jpeg", img), ai.Text("What is in this picture?"))
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(resp)
		checkMatch(t, resp.Results()[0], "man|person", "computer|laptop")
	}
	testFunctionCall(t, os.Getenv("ANTHROPIC_MODEL_FOR_TOOLS"), anthropic)
}
