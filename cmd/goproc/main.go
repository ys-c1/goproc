package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: goproc <gen|normalreport> [flags]")
		os.Exit(2)
	}
	var code int
	switch os.Args[1] {
	case "gen":
		code = runGen(os.Args[2:])
	case "normalreport":
		code = runNormalReport(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n", os.Args[1])
		fmt.Fprintln(os.Stderr, "usage: goproc <gen|normalreport> [flags]")
		code = 2
	}
	os.Exit(code)
}
