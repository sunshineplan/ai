package ai

import (
	"encoding/base64"
	"io"
	"net/http"
	"net/url"
	"strings"
)

type Content struct {
	Parts []Part
	Role  string
}

type Part interface {
	implementsPart()
}

type Text string

func (Text) implementsPart() {}

type Blob struct {
	MIMEType string
	Data     []byte
}

func (Blob) implementsPart() {}

type Image string

func ImageData(mime string, data []byte) Image {
	return Image("data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data))
}

func (Image) implementsPart() {}

func (img Image) MIMEType() string {
	u, err := url.Parse(string(img))
	if err != nil {
		panic(err)
	}
	switch u.Scheme {
	case "data":
		mime, _, _ := strings.Cut(u.Opaque, ";")
		return mime
	case "http", "https":
		resp, err := http.Head(u.String())
		if err != nil {
			panic(err)
		}
		defer resp.Body.Close()
		return resp.Header.Get("Content-Type")
	default:
		panic("unsupported image scheme: " + u.Scheme)
	}
}

func (img Image) Data() (mime string, data []byte) {
	u, err := url.Parse(string(img))
	if err != nil {
		panic(err)
	}
	switch u.Scheme {
	case "data":
		mime, b64, _ := strings.Cut(u.Opaque, ";base64,")
		b, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			panic(err)
		}
		return mime, b
	case "http", "https":
		resp, err := http.Get(u.String())
		if err != nil {
			panic(err)
		}
		defer resp.Body.Close()
		b, err := io.ReadAll(resp.Body)
		if err != nil {
			panic(err)
		}
		return resp.Header.Get("Content-Type"), b
	default:
		panic("unsupported image scheme: " + u.Scheme)
	}
}

type FunctionCall struct {
	ID        string
	Name      string
	Arguments string
}

func (FunctionCall) implementsPart() {}

type FunctionResponse struct {
	ID       string
	Response string
}

func (FunctionResponse) implementsPart() {}
