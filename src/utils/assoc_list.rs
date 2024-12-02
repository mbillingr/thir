use std::borrow::Borrow;
use std::ops::Index;
use crate::utils::lists::List;


#[derive(Clone, Debug)]
pub struct AssocList<K, V>(List<(K, V)>);


impl<K, V> Default for AssocList<K, V> {
    fn default() -> Self {
        AssocList(List::default())
    }
}


impl<K: PartialEq, V> AssocList<K, V> {

    pub fn insert(&mut self, key: K, value: V) {
        self.0 = self.0.cons((key, value))
    }
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where K: Borrow<Q>, Q: PartialEq
    {
        self.0.iter().find_map(|(k, v)| {
            if k.borrow() == key {
                Some(v)
            } else {
                None
            }
        })
    }
}

impl<K, Q, V> Index<&Q> for AssocList<K, V> where K: PartialEq + Borrow<Q>, Q: PartialEq {
    type Output = V;

    fn index(&self, index: &Q) -> &Self::Output {
        self.get(&index).expect("Key not found")
    }
}