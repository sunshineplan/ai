package ai

import (
	"encoding/json"
	"testing"
)

func TestLLMs(t *testing.T) {
	var llms LLMs
	if err := json.Unmarshal([]byte(`"chatgpt"`), &llms); err != nil {
		t.Error(err)
	} else if llms != ChatGPT {
		t.Errorf("expected %s; got %s", ChatGPT, llms)
	}
	if err := json.Unmarshal([]byte(`""`), &llms); err == nil {
		t.Error("expected error; got nil")
	}
	if err := json.Unmarshal([]byte(`"test"`), &llms); err == nil {
		t.Error("expected error; got nil")
	}
}
