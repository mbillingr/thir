# Typing Haskell in Rust

This is a bare-bones type checker for a Haskell-like type system.
Essentially, it is a Rust implementation
of [Typing Haskell in Haskell](https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA).

In this branch, we're deviating a lot from Haskell. The aim is to create build a somewhat usable language around a core
made of Haskell's type system.

Goals:

- Type inference, type classes, algebraic data types, polymorphic functions
- Eager evaluation
- Mutable state possible
