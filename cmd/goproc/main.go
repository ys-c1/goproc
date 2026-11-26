// Command goproc 提供程序化地形生成与 LOD 接缝检查。
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: goproc gen|lodcheck [flags]")
		os.Exit(2)
	}
	var code int
	switch os.Args[1] {
	case "gen":
		code = runGen(os.Args[2:])
	case "lodcheck":
		code = runLodCheck(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Println("usage: goproc gen|lodcheck [flags]")
		code = 0
	default:
		fmt.Fprintln(os.Stderr, "unknown subcommand:", os.Args[1])
		fmt.Fprintln(os.Stderr, "usage: goproc gen|lodcheck [flags]")
		code = 2
	}
	os.Exit(code)
}
