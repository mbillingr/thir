use crate::assumptions::{find, Assump};
use crate::classes::ClassEnv;
use crate::kinds::HasKind;
use crate::scheme::Scheme;
use crate::specific_inference::{ti_expl, ti_program, Alt, BindGroup, Expl, Program};
use crate::substitutions::Types;
use crate::type_inference::TI;
use crate::types::{Type, Tyvar};
use crate::Id;

#[derive(Debug)]
pub struct MethodImpl {
    pub method: Id,
    pub ty: Type,
    pub alts: Vec<Alt>,
}

#[derive(Debug)]
pub struct Module {
    pub impls: Vec<MethodImpl>,
    pub free_bindings: BindGroup,
}

pub fn ti_module(
    ce: &ClassEnv,
    ass: &[Assump],
    Module {
        impls,
        free_bindings,
    }: &Module,
) -> crate::Result<Module> {
    let pg = Program(vec![free_bindings.clone()]);
    let (bg_, mut ass_) = match ti_program(ce, ass.to_vec(), &pg)? {
        (Program(mut bgs), ass_) => (bgs.pop().unwrap(), ass_),
    };
    ass_.extend(ass.into_iter().cloned());

    let mut impls_ = vec![];
    for MethodImpl { method, ty, alts } in impls {
        let (mut ti, sc) = method_context(&ty, find(method, ass)?)?;

        let expl = Expl(method.clone(), sc, alts.clone());
        let (Expl(_, _, alts_), _) = ti_expl(&mut ti, ce, ass, &expl)?;

        println!("{:?}", alts_);
        impls_.push(MethodImpl {
            method: method.clone(),
            ty: ty.clone(),
            alts: alts_,
        })
    }

    Ok(Module {
        impls: impls_,
        free_bindings: bg_,
    })
}

/// Partially instantiate the first type variable in a type scheme.
/// The scheme is assumed to be a method signature, and the type is assumed to be the
/// argument to the class's type parameter.
/// The result can be used to check the method body.
fn method_context(ty: &Type, sc: &Scheme) -> Result<(TI, Scheme), String> {
    let mut ti = TI::new();

    let q = ti.fresh_inst(sc);

    // This is super hacky and relies on the following assumptions:
    //   - the first fresh variable is named `v0`
    //   - the first fresh variable created by fresh_inst above is the class type parameter
    //   - subsequent unifications will detect if ty has the wrong kind, which we simply force here
    ti.unify(&Type::TVar(Tyvar("v0".into(), ty.kind()?.clone())), ty)?;
    let q_ = ti.get_subst().apply(&q);

    let sc = Scheme::quantify(&q.tv(), &q_);

    Ok((ti, sc))
}
