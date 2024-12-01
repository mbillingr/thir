
// function composition operator
(.) : forall a b c => (b -> c) -> (a -> b) -> a -> c;
(.) f g x = f (g x);

//println : forall (a : Show) => a -> ();
println x = { puts . show x; puts "\n" };
