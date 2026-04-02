# Typing Haskell in Rust
This is a bare-bones type checker for a Haskell-like type system.
Essentially, it is a Rust implementation of [Typing Haskell in Haskell](https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA).

## Note to Future Me
When you consider working on this once more, the hard part is adding 
user-definable type classes and class implementations.

The type system as implemented type checks a whole program, assuming that
all type classes and other values are already defined. 
Adding a new type class is easy enough, so is adding a method.
However, defining the interface for a type class and implementing it
for different types is not part of this type checker. 
This is where I got stuck the first time, and this time again. 
