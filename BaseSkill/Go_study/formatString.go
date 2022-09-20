package main

import (
	"fmt"
)

func mai2n() {
	var stockcode=123
	var enddate="2020"
	var url="Code=%d&endDate=%s"
	var target_url=fmt.Sprintf(url,stockcode,enddate)
	fmt.Println(target_url)
}