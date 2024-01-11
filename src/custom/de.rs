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
    use crate::thir_core::predicates::Pred;
    use crate::thir_core::qualified::Qual;
    use crate::thir_core::scheme::Scheme;
    use crate::thir_core::scheme::Scheme::Forall;
    use crate::thir_core::types::{Tycon, Type, Tyvar};
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
                "interface Foo ( Bar Baz ) { foo forall [ ] [ ] TCon bla * }"
            )
            .unwrap(),
            ast::Interface {
                name: ast::Id::new("Foo"),
                supers: vec![ast::Id::new("Bar"), ast::Id::new("Baz")],
                methods: map![ast::Id::new("foo") => Forall(vec![], Qual(vec![], Type::TCon(Tycon("bla".into(), Star))))],
            }
        );
    }

    #[test]
    fn implementation() {
        assert_eq!(
            serde_json::from_str::<ast::Implementation>(
                "{\"name\": \"Foo\", \"for\": { \"TCon\": [\"bla\", \"*\"] }, \"preds\": [], \"methods\": {}}"
            )
            .unwrap(),
            ast::Implementation {
                name: ast::Id::new("Foo"),
                ty: Type::TCon(Tycon("bla".into(), Star)),
                preds: vec![],
                methods: Map::default(),
            }
        );

        let foo = ast::Id::new("Foo");
        let bar = ast::Id::new("bar");
        let x = ast::Id::new("x");
        assert_eq!(
            serde_src::from_str::<ast::Implementation>(
                "\
            implementation Foo TVar a * [  \
                IsIn Baz TVar a *   \
            ] {  \
                bar [  \
                    [ pvar x ] var x  \
                ]  \
            }"
            )
            .unwrap(),
            ast::Implementation {
                name: foo,
                ty: Type::TVar(Tyvar("a".into(), Star)),
                preds: vec![Pred::IsIn(
                    "Baz".into(),
                    Type::TVar(Tyvar("a".into(), Star))
                )],
                methods: map![bar => vec![ast::Alt(vec![ast::Pat::PVar(x.clone())], ast::Expr::Var(x))]],
            }
        );
    }
}
