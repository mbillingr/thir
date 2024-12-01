
// although the language provides syntactic sugar for lists,
// the have to be explicitly defined in some way.
data [] a = Nil | (::) a [a];

// TODO
//   Problem 1: using a instead of b below conflicts with another a (presumably the one in the class def)

impl Show forall (b : Show) => [b] {
    show xs = let commasep (Nil) = ""
                         | (x :: (Nil)) = show x
                         | (x :: xs) = (show x) ++ ", " ++ (commasep xs)
              in "[" ++ (commasep xs) ++ "]";
}


// function composition operator
(.) : forall a b c => (b -> c) -> (a -> b) -> a -> c;
(.) f g x = f (g x);

//println : forall (a : Show) => a -> ();
println x = { puts . show x; puts "\n" };
