package ai

import (
	"time"

	"golang.org/x/time/rate"
)

type Limiter interface {
	SetLimit(rate.Limit)
}

func NewLimiter(limit rate.Limit) *rate.Limiter {
	if limit == rate.Inf {
		return nil
	}
	return rate.NewLimiter(rate.Every(time.Minute)*limit, int(limit))
}
