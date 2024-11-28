
data MyList a = MyNil | MyCons a (MyList a);

nil = MyNil;
cons head tail = MyCons head tail;

head (MyCons x _) = x;
tail (MyCons _ xs) = xs;

map : forall a b => (a -> b) -> (MyList a) -> (MyList b);
map f (MyNil) = MyNil
  | f (MyCons x xs) = MyCons (f x) (map f xs);

