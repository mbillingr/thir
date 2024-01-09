# Typing Haskell in Rust
This is a bare-bones type checker for a Haskell-like type system.
Essentially, it is a Rust implementation of [Typing Haskell in Haskell](https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA).

This branch attempts to expose the core algorithm by stripping away complex features:
- ambiguity resolution (not really a must-have)
- kinds (what did we lose?)
