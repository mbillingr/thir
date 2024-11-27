# Typing Haskell in Rust

This is a bare-bones type checker for a Haskell-like type system.
Essentially, it is a Rust implementation
of [Typing Haskell in Haskell](https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA).

The current branch adds a REPL to make interaction with the system easier.

It's possible to define interfaces (data classes) and implementations, new datatypes and functions.
Everything is type-checked but no interpretation is performed at the moment. To get some form of feedback, the REPL
allows inspecting the environment, or show the type of an expression at the top level.

Syntactically, the language loosely resembles a mix of Haskell, Idris, ML and whatever seemed convenient to me at the
moment. It's not yet documented; you'll have to read the [parser's grammar](src/grammar.lalrpop) definition. Sorry :(
