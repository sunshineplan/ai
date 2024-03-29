package gemini

import "net/http"

var _ http.RoundTripper = new(apikey)

type apikey struct {
	key string
	rt  http.RoundTripper
}

func (t *apikey) RoundTrip(req *http.Request) (*http.Response, error) {
	args := req.URL.Query()
	args.Set("key", t.key)
	req.URL.RawQuery = args.Encode()
	return t.rt.RoundTrip(req)
}
