# Typing Haskell in Rust
This is a bare-bones type checker for a Haskell-like type system.
Essentially, it is a Rust implementation of [Typing Haskell in Haskell](https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA).

This branch explores an extension that allows type classes to have multiple type parameters.
(I.e. the traditional type classes are implemented for *a* type. Here we implement them for a combination of types.)