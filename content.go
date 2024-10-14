package ai

type Content struct {
	Parts []Part
	Role  string
}

type Part interface {
	implementsPart()
}

type Text string

func (t Text) implementsPart() {}

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

func (t Image) implementsPart() {}
