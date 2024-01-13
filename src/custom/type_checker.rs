use crate::custom::ast;
use crate::custom::ast::Alt;
use crate::custom::persistent::PersistentMap;
use crate::thir_core::assumptions::Assump;
use crate::thir_core::classes::{ClassEnv, EnvTransformer};
use crate::thir_core::kinds::Kind::Star;
use crate::thir_core::lists::ListLike;
use crate::thir_core::predicates::Pred;
use crate::thir_core::qualified::Qual;
use crate::thir_core::scheme::Scheme;
use crate::thir_core::Id;

type Result<T> = std::result::Result<T, String>;

/// Wrap thih's class env and add additional info
#[derive(Default)]
pub struct InterfaceEnv {
    ce: ClassEnv,
    method_impls: PersistentMap<Id, PersistentMap<Id, Vec<Alt>>>,
}

fn check_toplevel(
    InterfaceEnv { ce, method_impls }: &InterfaceEnv,
    top: &ast::Toplevel,
) -> Result<ast::Toplevel> {
    // interface definitions
    let ce = top
        .interface_defs
        .iter()
        .map(|intf| EnvTransformer::add_class(intf.name.clone(), intf.supers.clone()))
        .reduce(EnvTransformer::compose)
        .map(|et| et.apply(ce))
        .transpose()?
        .unwrap_or_else(|| ce.clone());

    // method declarations
    let ass: Vec<_> = top
        .interface_defs
        .iter()
        .flat_map(|intf| intf.methods.iter())
        .map(|(i, sc)| match sc {
            Scheme::Forall(ks, Qual(ps, t)) => Assump {
                i: i.clone(),
                // todo: currently, we assume Kind * for te interface type. this is an unnecessary limitation, but
                //       we might have to add the kind explicitly to the interface definition.
                sc: Scheme::Forall(
                    ks.cons(Star),
                    Qual(ps.snoc(todo!("this interface as predicate")), t.clone()),
                ),
            },
        })
        .collect();

    // interface implementations
    let ce = top
        .interface_impls
        .iter()
        .map(|imp| {
            EnvTransformer::add_inst(
                imp.preds.clone(),
                Pred::IsIn(imp.name.clone(), imp.ty.clone()),
            )
        })
        .reduce(EnvTransformer::compose)
        .map(|et| et.apply(&ce))
        .transpose()?
        .unwrap_or_else(|| ce.clone());

    let method_impls: Vec<_> = top
        .interface_impls
        .iter()
        .map(|imp| imp.methods.iter().map(|(method, expr)| method))
        .flatten()
        .collect();

    println!("{:?}", ce);
    println!("{:?}", ass);

    todo!("add (type-checked?) method impls to Inst?");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::serde_src;

    #[test]
    fn toplevel() {
        let ie = InterfaceEnv::default();
        let top = serde_src::from_str(
            "toplevel [
            interface Foo [ ] {
                 foo forall [ ] [  ] TCon Int * 
            }
        ] [
            impl Foo TCon Int * [ ] { }
        ]",
        )
        .unwrap();
        check_toplevel(&ie, &top).unwrap();
    }
}
