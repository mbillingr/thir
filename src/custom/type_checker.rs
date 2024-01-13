use crate::custom::ast;
use crate::custom::ast::Alt;
use crate::custom::persistent::{PersistentMap as Map, PersistentMap};
use crate::thir_core::assumptions::Assump;
use crate::thir_core::classes::{ClassEnv, EnvTransformer};
use crate::thir_core::kinds::Kind::Star;
use crate::thir_core::lists::ListLike;
use crate::thir_core::predicates::Pred;
use crate::thir_core::qualified::Qual;
use crate::thir_core::scheme::Scheme;
use crate::thir_core::type_inference::TI;
use crate::thir_core::types::Type;
use crate::thir_core::types::Type::TGen;
use crate::thir_core::{Id, Int};

type Result<T> = std::result::Result<T, String>;

/// Wrap thih's class env and add additional info
#[derive(Default)]
pub struct InterfaceEnv {
    ce: ClassEnv,
    method_impls: Map<Id, Map<Id, Vec<Alt>>>,
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
    let assumptions: Map<Id, Scheme> = top
        .interface_defs
        .iter()
        .map(|intf| {
            intf.methods.iter().map(|(i, sc)| match sc {
                Scheme::Forall(ks, Qual(ps, t)) => (
                    i.clone(),
                    Scheme::Forall(
                        // todo: currently, we assume Kind * for te interface type. this is an unnecessary limitation,
                        //       but we might have to add the kind explicitly to the interface definition.
                        ks.cons(Star),
                        Qual(ps.snoc(Pred::IsIn(intf.name.clone(), TGen(0))), t.clone()),
                    ),
                ),
            })
        })
        .flatten()
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
        .map(|imp| {
            imp.methods
                .iter()
                .map(|(method, alts)| check_method(&ce, &assumptions, method, alts))
        })
        .flatten()
        .collect::<Result<_>>()?;

    println!("{:?}", ce);
    println!("{:?}", assumptions);

    todo!("add (type-checked?) method impls to Inst?");
}

fn check_method(
    ce: &ClassEnv,
    ass: &PersistentMap<Id, Scheme>,
    method: &Id,
    alts: &[Alt],
) -> Result<Vec<Alt>> {
    let sc = ass
        .get(method)
        .ok_or_else(|| format!("Unknown method {method}"))?;

    let mut ti = TI::new();

    let Qual(qs, t) = ti.fresh_inst(sc);

    let (ps, alts_) = check_alts(&mut ti, ce, ass, alts, &t)?;
    todo!()
}

fn check_alts(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &PersistentMap<Id, Scheme>,
    alts: &[Alt],
    t: &Type,
) -> Result<(Vec<Pred>, Vec<Alt>)> {
    todo!()
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
            impl Foo TCon Int * [ ] {
                foo [ [ PVar x ] Var x 
                      [ PVar x ] Var x
                    ]
            }
        ]",
        )
        .unwrap();
        check_toplevel(&ie, &top).unwrap();
    }
}
