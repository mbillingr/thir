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
    use crate::custom::serde_src;
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
            serde_src::from_str::<ast::Interface>("interface Foo ( ) { }").unwrap(),
            ast::Interface {
                name: ast::Id::new("Foo"),
                supers: vec![],
                methods: Map::default(),
            }
        );
    }
}
