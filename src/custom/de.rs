use crate::custom::ast;
use crate::custom::persistent::PersistentMap as Map;
use serde::{Deserialize, Deserializer};

/*impl<'de> Deserialize<'de> for ast::Id {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::persistent::PersistentMap;
    use crate::custom::serde_src;
    use crate::map;
    use crate::thir_core::kinds::Kind::Star;
    use crate::thir_core::qualified::Qual;
    use crate::thir_core::scheme::Scheme;
    use crate::thir_core::scheme::Scheme::Forall;
    use crate::thir_core::types::{Tycon, Type};
    use serde_json;

    #[test]
    fn id() {
        assert_eq!(
            serde_json::from_str::<ast::Id>("\"foo-bar\"").unwrap(),
            ast::Id::new("foo-bar")
        );

        assert_eq!(
            serde_src::from_str::<ast::Id>("foo-bar").unwrap(),
            ast::Id::new("foo-bar")
        );
    }

    #[test]
    fn interface() {
        assert_eq!(
            serde_json::from_str::<ast::Interface>(
                "{\"name\": \"Foo\", \"supers\": [], \"methods\": {}}"
            )
            .unwrap(),
            ast::Interface {
                name: ast::Id::new("Foo"),
                supers: vec![],
                methods: Map::default(),
            }
        );

        assert_eq!(
            serde_json::from_str::<ast::Interface>("[\"Foo\", [], {}]").unwrap(),
            ast::Interface {
                name: ast::Id::new("Foo"),
                supers: vec![],
                methods: Map::default(),
            }
        );

        assert_eq!(
            serde_src::from_str::<ast::Interface>(
                "interface Foo ( Bar Baz ) { foo Forall [ ] [ [ ] TCon [ bla Star ] ] }"
            )
            .unwrap(),
            ast::Interface {
                name: ast::Id::new("Foo"),
                supers: vec![ast::Id::new("Bar"), ast::Id::new("Baz")],
                methods: map![ast::Id::new("foo") => Forall(vec![], Qual(vec![], Type::TCon(Tycon("bla".into(), Star))))],
            }
        );
    }
}
