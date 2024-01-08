/*!
Implementation of persistent maps with hash array mapped tries.

Uses reference counting to share structure. This may not be the most efficient way.
!*/

mod builder_macros;
mod hamt;
mod persistent_map;
mod persistent_set;
mod trie;

use hamt::Hamt;
pub use persistent_map::PersistentMap;
pub use persistent_set::PersistentSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use trie::Trie;

const NODE_ARRAY_SIZE: usize = 32;
const NODE_ARRAY_BITS: u32 = NODE_ARRAY_SIZE.ilog2();
const NODE_ARRAY_MASK: u64 = NODE_ARRAY_SIZE as u64 - 1;

enum RemoveResult<K, T> {
    NotFound,
    Removed,
    Replaced(Trie<K, T>),
}

fn split<K: Eq + Hash, T>(leaf1: Trie<K, T>, k1: u64, leaf2: Trie<K, T>, k2: u64) -> Trie<K, T> {
    if k1 == 0 && k2 == 0 {
        todo!("ran out of hash bits")
    }
    let idx1 = k1 & NODE_ARRAY_MASK;
    let mb1 = 1 << idx1;

    let idx2 = k2 & NODE_ARRAY_MASK;
    let mb2 = 1 << idx2;

    if idx1 == idx2 {
        return Trie::Node(Hamt::new(
            mb1,
            vec![split(
                leaf1,
                k1 >> NODE_ARRAY_BITS,
                leaf2,
                k2 >> NODE_ARRAY_BITS,
            )],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map;

    #[test]
    fn insert_and_retrieve() {
        let a = map![];
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

        let map = map!["x" => 1, "y" => 2];
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
        let a = map!["a" => 1, "b" => 2];
        let b = map!["b" => 3, "c" => 4];
        let e = map!["a" => 1, "b" => 3, "c" => 4];
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
        let a = map!["a" => 1, "b" => 2];
        let b = map!["b" => "x", "c" => "y"];
        let e = map!["b" => 2];
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
        let a = map!["a" => 1, "b" => 2];
        let b = map!["b" => "x", "c" => "y"];
        let e = map!["a" => 1];
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
        let a = map!["a" => 1, "b" => 2];
        let b = map!["b" => 3, "c" => 4];
        let e = map!["a" => 1, "c" => 4];
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
