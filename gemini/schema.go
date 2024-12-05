package gemini

import (
	"encoding/json"
	"strings"

	"github.com/google/generative-ai-go/genai"
)

type schema struct {
	Type        string
	Format      string
	Description string
	Nullable    bool
	Enum        []string
	Items       *schema
	Properties  map[string]*schema
	Required    []string
}

func (s schema) ToGenai() *genai.Schema {
	p := make(map[string]*genai.Schema)
	for k, v := range s.Properties {
		p[k] = v.ToGenai()
	}
	var items *genai.Schema
	if s.Items != nil {
		items = s.Items.ToGenai()
	}
	return &genai.Schema{
		Type:        genaiType(s.Type),
		Format:      s.Format,
		Description: s.Description,
		Nullable:    s.Nullable,
		Enum:        s.Enum,
		Items:       items,
		Properties:  p,
		Required:    s.Required,
	}
}

func genaiType(t string) genai.Type {
	switch strings.ToLower(t) {
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	case "object":
		return genai.TypeObject
	default:
		return genai.TypeUnspecified
	}
}

func genaiProperties(m map[string]any) (map[string]*genai.Schema, error) {
	b, _ := json.Marshal(m)
	var p map[string]schema
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, err
	}
	pp := make(map[string]*genai.Schema)
	for k, v := range p {
		pp[k] = v.ToGenai()
	}
	return pp, nil
}
