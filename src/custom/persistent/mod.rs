/*!
Implementation of persistent maps with hash array mapped tries.

Uses reference counting to share structure. This may not be the most efficient way.
!*/

mod hamt;
mod trie;

use hamt::Hamt;
use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use trie::Trie;

#[derive(Debug, PartialEq)]
pub struct PersistentMap<K, T>(Hamt<K, T>);

impl<K: Hash + Eq, T> PersistentMap<K, T> {
    #[inline]
    pub fn new() -> Self {
        PersistentMap(Hamt::new(0, vec![]))
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_pair(key).map(|(_, v)| v)
    }

    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_pair(key).is_some()
    }

    #[inline]
    pub fn get_pair<Q: ?Sized>(&self, key: &Q) -> Option<(&K, &T)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k = hash(&key);
        self.0.get(key, k).map(|rc| (&rc.0, &rc.1))
    }

    #[inline]
    pub fn insert(&self, key: K, val: T) -> Self {
        let k = hash(&key);
        let hamt = self.0.insert(key, val, k);
        PersistentMap(hamt)
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&self, key: &Q) -> Option<Self>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k = hash(&key);
        self.0.remove_from_root(key, k).map(Self)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Merge two maps.
    /// Returns a new map that contains all entries from two maps.
    /// If a key is present in both maps, the value is taken from `other`.
    #[inline]
    pub fn merge(&self, other: &Self) -> Self {
        PersistentMap(self.0.merge(&other.0))
    }

    /// Set intersection on map keys.
    /// That is, returns a new map that contains all keys that are also in another map.
    #[inline]
    pub fn intersect<U>(&self, other: &PersistentMap<K, U>) -> Self {
        PersistentMap(self.0.intersect(&other.0))
    }

    /// Set difference on map keys.
    /// That is, returns a new map that contains all keys that are not contained in another map.
    #[inline]
    pub fn difference<U>(&self, other: &PersistentMap<K, U>) -> Self {
        PersistentMap(self.0.difference(&other.0))
    }

    /// Symmetric set difference on map keys.
    /// That is, returns a new map that contains all keys that are in either map but not in both.
    #[inline]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        PersistentMap(self.0.symmetric_difference(&other.0))
    }
}

impl<K, T> Clone for PersistentMap<K, T> {
    fn clone(&self) -> Self {
        PersistentMap(self.0.clone())
    }
}

#[derive(Debug)]
pub struct PersistentSet<T>(PersistentMap<T, ()>);

impl<T: Hash + Eq> PersistentSet<T> {
    #[inline]
    pub fn new() -> Self {
        PersistentSet(PersistentMap::new())
    }

    #[inline]
    pub fn contains<Q: ?Sized>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.get_pair(key).is_some()
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.get_pair(key).map(|(k, _)| k)
    }

    #[inline]
    pub fn insert(&self, key: T) -> Self {
        PersistentSet(self.0.insert(key, ()))
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&self, key: &Q) -> Option<Self>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.remove(key).map(Self)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        PersistentSet(self.0.merge(&other.0))
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        PersistentSet(self.0.intersect(&other.0))
    }

    #[inline]
    pub fn difference(&self, other: &Self) -> Self {
        PersistentSet(self.0.difference(&other.0))
    }

    #[inline]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        PersistentSet(self.0.symmetric_difference(&other.0))
    }
}

const LEAF_SIZE: usize = 32;
const LEAF_BITS: u32 = LEAF_SIZE.ilog2();
const LEAF_MASK: u64 = LEAF_SIZE as u64 - 1;

enum RemoveResult<K, T> {
    NotFound,
    Removed,
    Replaced(Trie<K, T>),
}

fn split<K: Eq + Hash, T>(leaf1: Trie<K, T>, k1: u64, leaf2: Trie<K, T>, k2: u64) -> Trie<K, T> {
    if k1 == 0 && k2 == 0 {
        todo!("ran out of hash bits")
    }
    let idx1 = k1 & LEAF_MASK;
    let mb1 = 1 << idx1;

    let idx2 = k2 & LEAF_MASK;
    let mb2 = 1 << idx2;

    if idx1 == idx2 {
        return Trie::Node(Hamt::new(
            mb1,
            vec![split(leaf1, k1 >> LEAF_BITS, leaf2, k2 >> LEAF_BITS)],
        ));
    }

    if mb1 < mb2 {
        Trie::Node(Hamt::new(mb1 | mb2, vec![leaf1, leaf2]))
    } else {
        Trie::Node(Hamt::new(mb1 | mb2, vec![leaf2, leaf1]))
    }
}

/// I think this hash function is not secure because it always generates the same hash on every run
fn hash(x: &(impl Hash + ?Sized)) -> u64 {
    let mut s = DefaultHasher::new();
    x.hash(&mut s);
    s.finish()
}

fn insert<T: Clone>(idx: usize, x: T, xs: &[T]) -> Vec<T> {
    let mut res = Vec::with_capacity(xs.len() + 1);
    let mut xs = xs.iter().cloned();
    for _ in 0..idx {
        res.push(xs.next().unwrap());
    }
    res.push(x);
    res.extend(xs);
    res
}

fn replace<T: Clone>(idx: usize, x: T, xs: &[T]) -> Vec<T> {
    let mut res = xs.to_vec();
    res[idx] = x;
    res
}

fn remove<T: Clone>(idx: usize, xs: &[T]) -> Vec<T> {
    let mut res = Vec::with_capacity(xs.len() - 1);
    let mut xs = xs.iter().cloned();
    for _ in 0..idx {
        res.push(xs.next().unwrap());
    }
    xs.next();
    res.extend(xs);
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_retrieve() {
        let a = PersistentMap::<&str, i8>::new();
        assert_eq!(a.get("x"), None);

        let b = a.insert("x", 1);
        assert_eq!(b.get(&"y"), None);
        assert_eq!(b.get(&"x"), Some(&1));

        let c = b.insert("y", 2);
        assert_eq!(c.get(&"y"), Some(&2));

        let z = c.insert("z", 0);
        let z = z.insert("zz", 0);
        let z = z.insert("zzz", 0);
        let z = z.insert("zzzz", 0);
        assert_eq!(z.get(&"zzzz"), Some(&0));
        let z = z.insert("zzzzz", 0);
        let z = z.insert("zzzzzz", 0);
        let z = z.insert("zzzzzzz", 0);
        assert_eq!(z.get(&"zzzzzzz"), Some(&0));
        assert_eq!(z.get(&"x"), Some(&1));
    }

    #[test]
    fn remove() {
        assert!(PersistentMap::<&str, i8>::new().remove("x").is_none());

        let map = PersistentMap::new().insert("x", 1).insert("y", 2);
        let a = map.remove("x").unwrap();
        assert_eq!(a.get("x"), None);
        assert_eq!(a.get("y"), Some(&2));
        assert_eq!(map.get("x"), Some(&1));

        assert_eq!(a.remove("y"), Some(PersistentMap::new()))
    }

    #[test]
    fn big_map() {
        let mut map = PersistentMap::new();
        for n in 0..10000 {
            map = map.insert(n, n);
        }
        for k in 0..10000 {
            assert_eq!(map.get(&k), Some(&k));
        }
        for k in 0..10000 {
            map = map.remove(&k).unwrap();
        }
        assert_eq!(map.len(), 0);
        assert_eq!(map, PersistentMap::new());
    }

    #[test]
    fn length() {
        let a = PersistentMap::new();
        assert_eq!(a.len(), 0);
        let b = a.insert(0, 1);
        assert_eq!(b.len(), 1);
        let c = b.insert(0, 0);
        assert_eq!(b.len(), 1);
        let mut d = c;
        for k in 0..1000 {
            d = d.insert(k, k)
        }
        assert_eq!(d.len(), 1000);
    }

    #[test]
    fn merge() {
        let a = PersistentMap::new().insert("a", 1).insert("b", 2);
        let b = PersistentMap::new().insert("b", 3).insert("c", 4);
        let e = PersistentMap::new()
            .insert("a", 1)
            .insert("b", 3)
            .insert("c", 4);
        assert_eq!(a.merge(&b), e);
    }

    #[test]
    fn merge_big() {
        let mut a = PersistentMap::new();
        for i in 0..2000 {
            a = a.insert(i, i);
        }
        let mut b = PersistentMap::new();
        for i in 1000..3000 {
            b = b.insert(i, -i);
        }
        let mut e = PersistentMap::new();
        for i in 0..3000 {
            e = e.insert(i, if i < 1000 { i } else { -i });
        }

        assert_eq!(a.merge(&b), e);
    }

    #[test]
    fn intersect() {
        let a = PersistentMap::new().insert("a", 1).insert("b", 2);
        let b = PersistentMap::new().insert("b", "x").insert("c", "y");
        let e = PersistentMap::new().insert("b", 2);
        assert_eq!(a.intersect(&b), e);
    }

    #[test]
    fn intersect_big() {
        let mut a = PersistentMap::new();
        for i in 0..2000 {
            a = a.insert(i, i);
        }
        let mut b = PersistentMap::new();
        for i in 1000..3000 {
            b = b.insert(i, -i);
        }
        let mut e = PersistentMap::new();
        for i in 1000..2000 {
            e = e.insert(i, i);
        }

        let c = a.intersect(&b);

        assert_eq!(c, e);
    }

    #[test]
    fn difference() {
        let a = PersistentMap::new().insert("a", 1).insert("b", 2);
        let b = PersistentMap::new().insert("b", "x").insert("c", "y");
        let e = PersistentMap::new().insert("a", 1);
        assert_eq!(a.difference(&a), PersistentMap::new());
        assert_eq!(a.difference(&b), e);
    }

    #[test]
    fn difference_big() {
        let mut a = PersistentMap::new();
        for i in 0..2000 {
            a = a.insert(i, i);
        }
        let mut b = PersistentMap::new();
        for i in 1000..3000 {
            b = b.insert(i, -i);
        }
        let mut e = PersistentMap::new();
        for i in 0..1000 {
            e = e.insert(i, i);
        }

        let c = a.difference(&b);

        assert_eq!(c, e);
    }

    #[test]
    fn symmetric_difference() {
        let a = PersistentMap::new().insert("a", 1).insert("b", 2);
        let b = PersistentMap::new().insert("b", 3).insert("c", 4);
        let e = PersistentMap::new().insert("a", 1).insert("c", 4);
        assert_eq!(a.symmetric_difference(&a), PersistentMap::new());
        assert_eq!(a.symmetric_difference(&b), e);
    }

    #[test]
    fn symmetric_difference_big() {
        let mut a = PersistentMap::new();
        for i in 0..2000 {
            a = a.insert(i, i);
        }
        let mut b = PersistentMap::new();
        for i in 1000..3000 {
            b = b.insert(i, i);
        }
        let mut e = PersistentMap::new();
        for i in 0..1000 {
            e = e.insert(i, i);
        }
        for i in 2000..3000 {
            e = e.insert(i, i);
        }

        let c = a.symmetric_difference(&b);

        assert_eq!(c, e);
    }
}
