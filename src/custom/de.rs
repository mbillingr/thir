use crate::custom::ast;
use crate::custom::persistent::PersistentMap as Map;
use serde::{Deserialize, Deserializer};

/*impl<'de> Deserialize<'de> for Id {
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
    use crate::thir_core::kinds::Kind::Star;
    use crate::thir_core::predicates::Pred;
    use crate::thir_core::qualified::Qual;
    use crate::thir_core::scheme::Scheme;
    use crate::thir_core::scheme::Scheme::Forall;
    use crate::thir_core::types::{Tycon, Type, Tyvar};
    use crate::thir_core::Id;
    use crate::{list, map};
    use serde_json;

    #[test]
    fn id() {
        assert_eq!(
            serde_json::from_str::<Id>("\"foo-bar\"").unwrap(),
            "foo-bar".into()
        );

        assert_eq!(
            serde_src::from_str::<Id>("foo-bar").unwrap(),
            "foo-bar".into()
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
                name: "Foo".into(),
                supers: list![],
                methods: Map::default(),
            }
        );

        assert_eq!(
            serde_json::from_str::<ast::Interface>("[\"Foo\", [], {}]").unwrap(),
            ast::Interface {
                name: "Foo".into(),
                supers: list![],
                methods: Map::default(),
            }
        );

        assert_eq!(
            serde_src::from_str::<ast::Interface>(
                "interface Foo ( Bar Baz ) { foo forall [ ] [ ] TCon bla * }"
            )
            .unwrap(),
            ast::Interface {
                name: "Foo".into(),
                supers: list!["Bar".into(), "Baz".into()],
                methods: map!["foo".into() => Forall(vec![], Qual(vec![], Type::TCon(Tycon("bla".into(), Star))))],
            }
        );
    }

    #[test]
    fn implementation() {
        assert_eq!(
            serde_json::from_str::<ast::Impl>(
                "{\"name\": \"Foo\", \"for\": { \"TCon\": [\"bla\", \"*\"] }, \"preds\": [], \"methods\": {}}"
            )
            .unwrap(),
            ast::Impl {
                name: "Foo".into(),
                ty: Type::TCon(Tycon("bla".into(), Star)),
                preds: vec![],
                methods: Map::default(),
            }
        );

        let foo = "Foo".into();
        let bar = "bar".into();
        assert_eq!(
            serde_src::from_str::<ast::Impl>(
                "\
            impl Foo TVar a * [  \
                IsIn Baz TVar a *   \
            ] {  \
                bar [  \
                    [ pvar x ] var x  \
                ]  \
            }"
            )
            .unwrap(),
            ast::Impl {
                name: foo,
                ty: Type::TVar(Tyvar("a".into(), Star)),
                preds: vec![Pred::IsIn(
                    "Baz".into(),
                    Type::TVar(Tyvar("a".into(), Star))
                )],
                methods: map![bar => vec![ast::Alt(vec![ast::Pat::PVar("x".into())], ast::Expr::Var("x".into()))]],
            }
        );
    }
}
