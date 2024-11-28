use crate::type_checker::instantiate::Instantiate;
use crate::type_checker::kinds::Kind;
use crate::type_checker::qualified::Qual;
use crate::type_checker::scheme::Scheme;
use crate::type_checker::substitutions::Subst;
use crate::type_checker::types::{Type, Tyvar};
use crate::type_checker::unification::mgu;
use crate::type_checker::GenId;
use crate::type_checker::Id;
use std::collections::HashMap;

/// The type inference state
pub struct TI {
    subst: Subst,
    count: GenId,
    annotations: HashMap<*const u8, Type>,
}

impl TI {
    pub fn new() -> Self {
        TI {
            subst: Subst::null_subst(),
            count: 0,
            annotations: Default::default(),
        }
    }

    pub fn get_subst(&self) -> &Subst {
        &self.subst
    }

    pub fn unify(&mut self, t1: &Type, t2: &Type) -> crate::Result<()> {
        let u = mgu(&self.subst.apply(t1), &self.subst.apply(t2))?;
        Ok(self.ext_subst(u))
    }

    pub fn new_tvar(&mut self, k: Kind) -> Type {
        let v = Tyvar(enum_id(self.count), k);
        self.count += 1;
        Type::TVar(v)
    }

    pub fn fresh_inst(&mut self, sc: &Scheme) -> Qual<Type> {
        match sc {
            Scheme::Forall(ks, qt) => {
                let ts: Vec<_> = ks.iter().map(|k| self.new_tvar(k.clone())).collect();
                qt.inst(&ts)
            }
        }
    }

    fn ext_subst(&mut self, s: Subst) {
        self.subst = s.compose(&self.subst); // todo: is the order of composition correct?
    }

    pub fn annotate<T>(&mut self, thing: &T, t: Type) {
        self.annotations.insert(thing as *const T as *const u8, t);
    }

    pub fn get_annotation<T>(&self, thing: &T) -> Option<Type> {
        self.annotations
            .get(&(thing as *const T as *const u8))
            .map(|t| self.subst.apply(t))
    }
}

fn enum_id(n: GenId) -> Id {
    format!("v{n}")
}
