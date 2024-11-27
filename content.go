package ai

type Content struct {
	Parts []Part
	Role  string
}

type Part interface {
	implementsPart()
}

type Text string

func (Text) implementsPart() {}

type Image struct {
	MIMEType string
	Data     []byte
}

func ImageData(mime string, data []byte) Image {
	return Image{
		MIMEType: mime,
		Data:     data,
	}
}

func (Image) implementsPart() {}

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
