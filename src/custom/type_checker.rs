use crate::custom::ast;
use crate::thir_core::assumptions::Assump;
use crate::thir_core::classes::{ClassEnv, EnvTransformer};
use crate::thir_core::predicates::Pred;

type Result<T> = std::result::Result<T, String>;

fn check_toplevel(ce: &ClassEnv, top: &ast::Toplevel) -> Result<ast::Toplevel> {
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
        .map(|(i, sc)| Assump {
            i: i.clone(),
            sc: sc.clone(),
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
        let ce = ClassEnv::default();
        let top = serde_src::from_str(
            "toplevel [
            interface Foo [ ] {
                 foo forall [ ] [ ] TCon Int * 
            }
        ] [
            impl Foo TCon Int * [ ] { }
        ]",
        )
        .unwrap();
        check_toplevel(&ce, &top).unwrap();
    }
}
