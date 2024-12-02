
// function composition operator
(.) : forall a b c => (b -> c) -> (a -> b) -> a -> c;
(.) f g x = f (g x);

//println : forall (a : Show) => a -> ();
println x = { puts . show x; puts "\n" };



impl Show forall (a : Show) => [a] {
    show xs = let commasep (Nil) = ""
                         | (x :: (Nil)) = show x
                         | (x :: xs) = (show x) ++ ", " ++ (commasep xs)
              in "[" ++ (commasep xs) ++ "]";
}

impl Concatenate forall b => [b] {
    (++)     (Nil) rhs = rhs
       | (x :: xs) rhs = x :: (xs ++ rhs);
}


interface Foldable f : * -> * {
    foldr : forall e a => (e -> a -> a) -> a -> f e -> a;
    foldl : forall e a => (a -> e -> a) -> a -> f e -> a;
}

impl Foldable for [] {
    foldr f init = let loop (Nil) = init
                          | (x :: xs) = f x (loop xs)
                   in loop;

    foldl f = let loop acc (Nil) = acc
                     | acc (x :: xs) = loop (f acc x) xs
              in loop;
}


interface Functor f : * -> * {
    map : forall u v => (u -> v) -> f u -> f v;
}

impl Functor for [] {
    map f = foldr ((::) . f) Nil;      
}

interface Filterable f : * -> * {
    filter : forall a => (a -> Bool) -> f a -> f a;
}

impl Filterable for [] {
    filter p = let loop (Nil) = Nil | (x :: xs) = if (p x) then x :: (loop xs) else (loop xs) in loop;
    
    // this is a bit slower:
    //filter p = foldr (fun x fxs = if (p x) then x :: fxs else fxs) Nil;
}


sort (Nil) = Nil
   | (x :: (Nil)) = [x]
   | (p :: xs) = (sort (filter (<= p) xs)) ++ [p] ++ (sort (filter (> p) xs))
;

