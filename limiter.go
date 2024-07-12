package ai

import (
	"math"
	"time"

	"golang.org/x/time/rate"
)

type Limiter interface {
	SetLimit(rpm int64)
	Limit() (rpm int64)
}

func NewLimiter(rpm int64) *rate.Limiter {
	if rpm == math.MaxInt64 {
		return nil
	}
	return rate.NewLimiter(rate.Every(time.Minute)*rate.Limit(rpm), int(rpm))
}
