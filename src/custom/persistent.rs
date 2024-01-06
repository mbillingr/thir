/*!
Implementation of persistent maps with hash array mapped tries.

Uses reference counting to share structure. This may not be the most efficient way.
!*/

use crate::custom::persistent::RemoveResult::{NotFound, Removed, Replaced};
use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

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
        let k = hash(&key);
        self.0.get(key, k)
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
}

#[derive(Debug)]
pub struct PersistentSet<T>(Hamt<T, ()>);

const LEAF_SIZE: usize = 32;
const LEAF_BITS: u32 = LEAF_SIZE.ilog2();
const LEAF_MASK: u64 = LEAF_SIZE as u64 - 1;

#[derive(Debug)]
enum Trie<K, T> {
    Leaf(Rc<(K, T)>),
    Node(Hamt<K, T>),
}

impl<K, T> Trie<K, T> {
    fn leaf(k: K, v: T) -> Self {
        Trie::Leaf(Rc::new((k, v)))
    }

    fn len(&self) -> usize {
        match self {
            Trie::Leaf(_) => 1,
            Trie::Node(hamt) => hamt.len(),
        }
    }
}
impl<K: Eq + Hash, T> Trie<K, T> {
    pub fn remove<Q: ?Sized>(&self, key: &Q, k: u64) -> RemoveResult<K, T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self {
            Trie::Leaf(rc) if rc.0.borrow() == key => Removed,
            Trie::Leaf(_) => return NotFound,
            Trie::Node(child) => child.remove_from_node(key, k >> LEAF_BITS),
        }
    }
}

impl<K, T> Clone for Trie<K, T> {
    fn clone(&self) -> Self {
        match self {
            Trie::Leaf(rc) => Trie::Leaf(rc.clone()),
            Trie::Node(hamt) => Trie::Node(hamt.clone()),
        }
    }
}

impl<K: PartialEq, T: PartialEq> PartialEq for Trie<K, T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Trie::Leaf(a), Trie::Leaf(b)) => a == b,
            (Trie::Node(a), Trie::Node(b)) => a == b,
            _ => false,
        }
    }
}

/// A Hash array mapped trie node
#[derive(Debug)]
struct Hamt<K, T> {
    mapping: u32,
    subtrie: Rc<[Trie<K, T>]>,
}

impl<K, T> Hamt<K, T> {
    fn new(mapping: u32, children: Vec<Trie<K, T>>) -> Self {
        Hamt {
            mapping,
            subtrie: children.into(),
        }
    }

    fn len(&self) -> usize {
        self.subtrie.iter().map(Trie::len).sum()
    }
}

impl<K: Eq + Hash, T> Hamt<K, T> {
    pub fn get<Q: ?Sized>(&self, key: &Q, k: u64) -> Option<&T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (mask_bit, idx_) = self.hash_location(k);

        if self.is_free(mask_bit) {
            return None;
        }

        match &self.subtrie[idx_] {
            Trie::Leaf(rc) if rc.0.borrow() == key => Some(&rc.1),
            Trie::Leaf(_) => None,
            Trie::Node(child) => child.get(key, k >> LEAF_BITS),
        }
    }
    #[inline]
    fn insert(&self, key: K, val: T, k: u64) -> Self {
        self._insert(key, val, k, LEAF_BITS)
    }

    fn _insert(&self, key: K, val: T, k: u64, depth: u32) -> Self {
        let (mask_bit, idx_) = self.hash_location(k);

        let subtrie = if self.is_free(mask_bit) {
            // free slot
            insert(idx_, Trie::leaf(key, val), &self.subtrie).into()
        } else {
            // occupied slot
            let new_child = match &self.subtrie[idx_] {
                Trie::Leaf(rc) if rc.0 == key => Trie::leaf(key, val),
                leaf @ Trie::Leaf(rc) => split(
                    Trie::leaf(key, val),
                    k >> LEAF_BITS,
                    leaf.clone(),
                    hash(&rc.0) >> depth,
                ),
                Trie::Node(child) => {
                    Trie::Node(child._insert(key, val, k >> LEAF_BITS, depth + LEAF_BITS))
                }
            };
            replace(idx_, new_child, &self.subtrie).into()
        };
        Hamt {
            mapping: self.mapping | mask_bit,
            subtrie,
        }
    }
    pub fn remove_from_root<Q: ?Sized>(&self, key: &Q, k: u64) -> Option<Self>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (mask_bit, idx_) = self.hash_location(k);

        if self.is_free(mask_bit) {
            return None;
        }

        // we always keep the root array, even if it's empty
        match self.subtrie[idx_].remove(key, k) {
            NotFound => None,
            Removed => Some(self.remove_child(mask_bit, idx_)),
            Replaced(new_child) => Some(self.replace_child(idx_, new_child)),
        }
    }
    pub fn remove_from_node<Q: ?Sized>(&self, key: &Q, k: u64) -> RemoveResult<K, T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (mask_bit, idx_) = self.hash_location(k);

        if self.is_free(mask_bit) {
            return NotFound;
        }

        match self.subtrie[idx_].remove(key, k) {
            NotFound => NotFound,

            Removed => match &*self.subtrie {
                [] => unreachable!(),
                // this non-root node just became empty, so we can remove it
                [_] => Removed,
                [_, _] => match &self.subtrie[1 - idx_] {
                    // if the only remaining child is a leaf we can replace this non-root node with it
                    leaf @ Trie::Leaf(_) => Replaced(leaf.clone()),
                    Trie::Node(_) => Replaced(Trie::Node(self.remove_child(mask_bit, idx_))),
                },
                _ => Replaced(Trie::Node(self.remove_child(mask_bit, idx_))),
            },

            Replaced(new_child) => Replaced(Trie::Node(self.replace_child(idx_, new_child))),
        }
    }

    #[inline]
    fn is_free(&self, mask_bit: u32) -> bool {
        self.mapping & mask_bit == 0
    }

    #[inline]
    fn hash_location(&self, k: u64) -> (u32, usize) {
        let idx = k & LEAF_MASK;
        let mask_bit: u32 = 1 << idx;
        let idx_ = u32::count_ones(self.mapping & (mask_bit - 1)) as usize;
        (mask_bit, idx_)
    }

    fn replace_child(&self, idx: usize, child: Trie<K, T>) -> Self {
        Hamt {
            mapping: self.mapping,
            subtrie: replace(idx, child, &self.subtrie).into(),
        }
    }

    fn remove_child(&self, mask_bit: u32, idx: usize) -> Self {
        Hamt {
            mapping: self.mapping & !mask_bit,
            subtrie: remove(idx, &self.subtrie).into(),
        }
    }
}

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

impl<K, T> Clone for Hamt<K, T> {
    fn clone(&self) -> Self {
        Hamt {
            mapping: self.mapping,
            subtrie: self.subtrie.clone(),
        }
    }
}

impl<K: PartialEq, T: PartialEq> PartialEq for Hamt<K, T> {
    fn eq(&self, other: &Self) -> bool {
        if self.mapping != other.mapping {
            return false;
        }

        self.subtrie
            .iter()
            .zip(other.subtrie.iter())
            .all(|(a, b)| a == b)
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
        println!("{z:?}");
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

        println!("{a:?}");
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
}
