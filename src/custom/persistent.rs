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

    #[inline]
    pub fn merge(&self, other: &Self) -> Self {
        PersistentMap(self.0.merge(&other.0, 0))
    }

    #[inline]
    pub fn intersect(&self, other: &Self) -> Self {
        PersistentMap(self.0.intersect(&other.0))
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
}

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

    fn is_leaf(&self) -> bool {
        match self {
            Trie::Leaf(_) => true,
            Trie::Node(_) => false,
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

    fn merge(&self, other: &Self, depth: u32) -> Self {
        match (self, other) {
            (Trie::Leaf(a), Trie::Leaf(b)) if a.0 == b.0 => other.clone(),
            (Trie::Leaf(a), Trie::Leaf(b)) => split(
                self.clone(),
                hash(&a.0) >> depth,
                other.clone(),
                hash(&b.0) >> depth,
            ),
            (Trie::Leaf(a), Trie::Node(b)) => {
                match b._insert(a.clone(), hash(&a.0) >> depth, depth + LEAF_BITS, false) {
                    None => other.clone(),
                    Some(b_) => Trie::Node(b_),
                }
            }
            (Trie::Node(a), Trie::Leaf(b)) => Trie::Node(
                a._insert(b.clone(), hash(&b.0) >> depth, depth + LEAF_BITS, true)
                    .unwrap(),
            ),
            (Trie::Node(a), Trie::Node(b)) => Trie::Node(a.merge(b, depth)),
        }
    }

    fn intersect(&self, other: &Self, depth: u32) -> Option<Self> {
        match (self, other) {
            (Trie::Leaf(a), Trie::Leaf(b)) if a.0 == b.0 => Some(other.clone()),
            (Trie::Leaf(_), Trie::Leaf(_)) => None,
            (Trie::Leaf(a), Trie::Node(b)) => {
                b.get(&a.0, hash(&a.0) >> depth).cloned().map(Trie::Leaf)
            }
            (Trie::Node(a), Trie::Leaf(b)) => {
                a.get(&b.0, hash(&b.0) >> depth).map(|_| other.clone())
            }
            (Trie::Node(a), Trie::Node(b)) => a._intersect(b, depth),
        }
    }

    fn show_diff(&self, other: &Self) {
        match (self, other) {
            (Trie::Leaf(a), Trie::Leaf(b)) => {
                if a.0 != b.0 {
                    println!("leafs differ")
                }
            }
            (Trie::Node(a), Trie::Node(b)) => {
                if a.mapping == b.mapping {
                    for (ax, bx) in a.subtrie.iter().zip(b.subtrie.iter()) {
                        ax.show_diff(bx)
                    }
                } else {
                    println!("{:32b}\n{:32b}\n", a.mapping, b.mapping)
                }
            }
            _ => todo!(),
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

    fn leaves(&self) -> LeafIterator<K, T> {
        LeafIterator::new(&self.subtrie)
    }
}

impl<K: Eq + Hash, T> Hamt<K, T> {
    pub fn get<Q: ?Sized>(&self, key: &Q, k: u64) -> Option<&Rc<(K, T)>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (mask_bit, idx_) = self.hash_location(k);

        if self.is_free(mask_bit) {
            return None;
        }

        match &self.subtrie[idx_] {
            Trie::Leaf(rc) if rc.0.borrow() == key => Some(rc),
            Trie::Leaf(_) => None,
            Trie::Node(child) => child.get(key, k >> LEAF_BITS),
        }
    }
    #[inline]
    fn insert(&self, key: K, val: T, k: u64) -> Self {
        self._insert(Rc::new((key, val)), k, LEAF_BITS, true)
            .unwrap()
    }

    fn _insert(
        &self,
        leaf: Rc<(K, T)>,
        k: u64,
        depth: u32,
        replace_existing: bool,
    ) -> Option<Self> {
        let (mask_bit, idx_) = self.hash_location(k);

        let subtrie = if self.is_free(mask_bit) {
            // free slot
            insert(idx_, Trie::Leaf(leaf), &self.subtrie).into()
        } else {
            // occupied slot
            let new_child = match &self.subtrie[idx_] {
                Trie::Leaf(rc) if rc.0 == leaf.0 => {
                    if replace_existing {
                        Trie::Leaf(leaf)
                    } else {
                        return None;
                    }
                }
                old_leaf @ Trie::Leaf(rc) => split(
                    Trie::Leaf(leaf),
                    k >> LEAF_BITS,
                    old_leaf.clone(),
                    hash(&rc.0) >> depth,
                ),
                Trie::Node(child) => Trie::Node(child._insert(
                    leaf,
                    k >> LEAF_BITS,
                    depth + LEAF_BITS,
                    replace_existing,
                )?),
            };
            replace(idx_, new_child, &self.subtrie).into()
        };
        Some(Hamt {
            mapping: self.mapping | mask_bit,
            subtrie,
        })
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

    fn merge(&self, other: &Self, depth: u32) -> Self {
        if self.ptr_eq(other) {
            return self.clone();
        }

        let mut mapping = 0;
        let mut subtrie = vec![];

        for ((m, t1), (m2, t2)) in self.slots().zip(other.slots()) {
            debug_assert_eq!(m, m2);
            match (t1, t2) {
                (None, None) => continue,
                (Some(a), None) => subtrie.push(a.clone()),
                (None, Some(b)) => subtrie.push(b.clone()),
                (Some(a), Some(b)) => subtrie.push(a.merge(b, depth + LEAF_BITS)),
            }
            mapping |= m;
        }
        Hamt {
            mapping,
            subtrie: subtrie.into(),
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        match self._intersect(other, 0) {
            None => Self::new(0, vec![]),
            Some(Trie::Node(hamt)) => hamt,
            Some(Trie::Leaf(rc)) => {
                let k = hash(&rc.0);
                Hamt::new(0, vec![])
                    ._insert(rc, k, LEAF_BITS, true)
                    .unwrap()
            }
        }
    }

    fn _intersect(&self, other: &Self, depth: u32) -> Option<Trie<K, T>> {
        if self.ptr_eq(other) {
            return Some(Trie::Node(self.clone()));
        }

        let mut mapping = 0;
        let mut subtrie = vec![];

        for ((m, t1), (m2, t2)) in self.slots().zip(other.slots()) {
            debug_assert_eq!(m, m2);
            match (t1, t2) {
                (None, _) => continue,
                (_, None) => continue,
                (Some(a), Some(b)) => match a.intersect(b, depth + LEAF_BITS) {
                    None => continue,
                    Some(t_) => subtrie.push(t_),
                },
            }
            mapping |= m;
        }

        match subtrie.len() {
            0 => None,
            1 if subtrie[0].is_leaf() => subtrie.pop(),
            _ => Some(Trie::Node(Hamt {
                mapping,
                subtrie: subtrie.into(),
            })),
        }
    }

    #[inline]
    fn ptr_eq(&self, other: &Self) -> bool {
        if Rc::ptr_eq(&self.subtrie, &other.subtrie) {
            debug_assert_eq!(self.mapping, other.mapping);
            true
        } else {
            false
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

    fn slots(&self) -> impl Iterator<Item = (u32, Option<&Trie<K, T>>)> {
        let mut children = self.subtrie.iter();
        (0..32).map(|idx| 1 << idx).map(move |bm| {
            (
                bm,
                if self.is_free(bm) {
                    None
                } else {
                    children.next()
                },
            )
        })
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

struct LeafIterator<'a, K, T> {
    stack: Vec<(usize, &'a Rc<[Trie<K, T>]>)>,
}

impl<'a, K, T> LeafIterator<'a, K, T> {
    pub fn new(root: &'a Rc<[Trie<K, T>]>) -> Self {
        LeafIterator {
            stack: vec![(0, root)],
        }
    }
}

impl<'a, K, T> Iterator for LeafIterator<'a, K, T> {
    type Item = &'a Rc<(K, T)>;

    fn next(&mut self) -> Option<Self::Item> {
        let (idx, arr) = self.stack.pop()?;
        if idx >= arr.len() {
            return self.next();
        }
        self.stack.push((idx + 1, arr));
        match &arr[idx] {
            Trie::Node(hamt) => {
                self.stack.push((0, &hamt.subtrie));
                self.next()
            }
            Trie::Leaf(rc) => Some(rc),
        }
    }
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
        let b = PersistentMap::new().insert("b", 3).insert("c", 4);
        let e = PersistentMap::new().insert("b", 3);
        assert_eq!(a.intersect(&b), e);
    }

    #[test]
    fn intersect_big() {
        let mut a = PersistentMap::new();
        for i in 0..2000 {
            a = a.insert(i, -i);
        }
        let mut b = PersistentMap::new();
        for i in 1000..3000 {
            b = b.insert(i, i);
        }
        let mut e = PersistentMap::new();
        for i in 1000..2000 {
            e = e.insert(i, i);
        }

        let c = a.intersect(&b);

        assert_eq!(c, e);
    }
}
