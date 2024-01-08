use crate::custom::persistent;
use crate::custom::persistent::trie::Trie;
use crate::custom::persistent::RemoveResult::{NotFound, Removed, Replaced};
use crate::custom::persistent::{RemoveResult, LEAF_BITS, LEAF_MASK};
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::rc::Rc;

/// A Hash array mapped trie node
pub struct Hamt<K, T> {
    mapping: u32,
    subtrie: Rc<[Trie<K, T>]>,
}

impl<K: Debug, T: Debug> Debug for Hamt<K, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hamt({:032b}: {:?}", self.mapping, self.subtrie)
    }
}

impl<K, T> Hamt<K, T> {
    pub fn new(mapping: u32, children: Vec<Trie<K, T>>) -> Self {
        Hamt {
            mapping,
            subtrie: children.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.subtrie.iter().map(Trie::len).sum()
    }

    pub fn leaves(&self) -> LeafIterator<K, T> {
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
    pub fn insert(&self, key: K, val: T, k: u64) -> Self {
        self._insert(Rc::new((key, val)), k, LEAF_BITS, true)
            .unwrap()
    }

    pub fn _insert(
        &self,
        leaf: Rc<(K, T)>,
        k: u64,
        depth: u32,
        replace_existing: bool,
    ) -> Option<Self> {
        let (mask_bit, idx_) = self.hash_location(k);

        let subtrie = if self.is_free(mask_bit) {
            // free slot
            persistent::insert(idx_, Trie::Leaf(leaf), &self.subtrie).into()
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
                old_leaf @ Trie::Leaf(rc) => persistent::split(
                    Trie::Leaf(leaf),
                    k >> LEAF_BITS,
                    old_leaf.clone(),
                    persistent::hash(&rc.0) >> depth,
                ),
                Trie::Node(child) => Trie::Node(child._insert(
                    leaf,
                    k >> LEAF_BITS,
                    depth + LEAF_BITS,
                    replace_existing,
                )?),
            };
            persistent::replace(idx_, new_child, &self.subtrie).into()
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
                [_] => Removed, // this non-root node just became empty, so we can remove it
                [_, _] if self.subtrie[1 - idx_].is_leaf() => {
                    // if the only remaining child is a leaf we can replace this non-root node with it
                    Replaced(self.subtrie[1 - idx_].clone())
                }
                _ => Replaced(Trie::Node(self.remove_child(mask_bit, idx_))),
            },

            Replaced(new_child) => {
                match &*self.subtrie {
                    [] => unreachable!(),
                    [_] if new_child.is_leaf() => {
                        // if the only remaining child is a leaf we can replace this non-root node with it
                        Replaced(new_child)
                    }
                    _ => Replaced(Trie::Node(self.replace_child(idx_, new_child))),
                }
            }
        }
    }

    pub fn merge(&self, other: &Self, depth: u32) -> Self {
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

    pub fn intersect<U>(&self, other: &Hamt<K, U>) -> Self {
        Self::make_root(self._intersect(other, 0))
    }

    pub fn _intersect<U>(&self, other: &Hamt<K, U>, depth: u32) -> Option<Trie<K, T>> {
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

    pub fn difference<U>(&self, other: &Hamt<K, U>) -> Self {
        Self::make_root(self._difference(other, 0))
    }

    pub fn _difference<U>(&self, other: &Hamt<K, U>, depth: u32) -> Option<Trie<K, T>> {
        if self.ptr_eq(other) {
            return None;
        }

        let mut mapping = 0;
        let mut subtrie = vec![];

        for ((m, t1), (m2, t2)) in self.slots().zip(other.slots()) {
            debug_assert_eq!(m, m2);
            match (t1, t2) {
                (Some(a), None) => subtrie.push(a.clone()),
                (None, _) => continue,
                (Some(a), Some(b)) => match a.difference(b, depth + LEAF_BITS) {
                    None => continue,
                    Some(t_) => subtrie.push(t_),
                },
            }
            mapping |= m;
        }

        let n = subtrie.len();
        match n {
            0 => None,
            1 if subtrie[0].is_leaf() => subtrie.pop(),
            _ => Some(Trie::Node(Hamt {
                mapping,
                subtrie: subtrie.into(),
            })),
        }
    }

    pub fn symmetric_difference(&self, other: &Self) -> Self {
        Self::make_root(self._symmetric_difference(other, 0))
    }

    pub fn _symmetric_difference(&self, other: &Self, depth: u32) -> Option<Trie<K, T>> {
        if self.ptr_eq(other) {
            return None;
        }

        let mut mapping = 0;
        let mut subtrie = vec![];

        for ((m, t1), (m2, t2)) in self.slots().zip(other.slots()) {
            debug_assert_eq!(m, m2);
            match (t1, t2) {
                (None, None) => continue,
                (Some(a), None) => subtrie.push(a.clone()),
                (None, Some(b)) => subtrie.push(b.clone()),
                (Some(a), Some(b)) => match a.symmetric_difference(b, depth + LEAF_BITS) {
                    None => continue,
                    Some(t_) => subtrie.push(t_),
                },
            }
            mapping |= m;
        }

        let n = subtrie.len();
        match n {
            0 => None,
            1 if subtrie[0].is_leaf() => subtrie.pop(),
            _ => Some(Trie::Node(Hamt {
                mapping,
                subtrie: subtrie.into(),
            })),
        }
    }

    fn make_root(trie: Option<Trie<K, T>>) -> Hamt<K, T> {
        match trie {
            None => Self::new(0, vec![]),
            Some(Trie::Node(hamt)) => hamt,
            Some(Trie::Leaf(rc)) => {
                let k = persistent::hash(&rc.0);
                Hamt::new(0, vec![])
                    ._insert(rc, k, LEAF_BITS, true)
                    .unwrap()
            }
        }
    }

    #[inline]
    fn ptr_eq<U>(&self, other: &Hamt<K, U>) -> bool {
        if Rc::as_ptr(&self.subtrie) as *const u8 == Rc::as_ptr(&other.subtrie) as *const u8 {
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
            subtrie: persistent::replace(idx, child, &self.subtrie).into(),
        }
    }

    fn remove_child(&self, mask_bit: u32, idx: usize) -> Self {
        Hamt {
            mapping: self.mapping & !mask_bit,
            subtrie: persistent::remove(idx, &self.subtrie).into(),
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

pub struct LeafIterator<'a, K, T> {
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
