package main

import (
	"fmt"
)

func main(){
	var a=0
	var b=0
	var output="%d*%d=%d"
	for a=0;a<10;a++{
		for b=a;b<10;b++{
			var res=fmt.Sprintf(output,a,b,a*b)
			fmt.Print(res)
			fmt.Print(" ")
		
		}
		fmt.Println(" ")
	}
}