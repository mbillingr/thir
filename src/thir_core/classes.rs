use crate::custom::persistent::PersistentMap;
use crate::list;
use crate::thir_core::lists::List;
use crate::thir_core::predicates::{match_pred, mgu_pred, Pred};
use crate::thir_core::qualified::Qual;
use crate::thir_core::types::Type;
use crate::thir_core::Id;
use std::rc::Rc;

/// A Type class (Interface) contains a list of super classes and a list of instances.
#[derive(Debug, Clone)]
pub struct Class(pub List<Id>, pub List<Inst>);

/// An instance is a type that implements a certain class (interface)
pub type Inst = Qual<Pred>;

/// The class environment captures information about defined classes and instances
/// in a given program.
#[derive(Clone, Debug)]
pub struct ClassEnv {
    classes: PersistentMap<Id, Class>,
    defaults: List<Type>,
}

impl Default for ClassEnv {
    fn default() -> Self {
        ClassEnv {
            classes: PersistentMap::new(),
            defaults: list![Type::t_int(), Type::t_double()],
        }
    }
}

impl ClassEnv {
    /// get super classes for a defined class
    pub fn supers(&self, name: &Id) -> List<Id> {
        self.classes.get(name).expect("class not defined").0.clone()
    }

    /// get instances for a defined class
    pub fn insts(&self, name: &Id) -> List<Inst> {
        self.classes.get(name).expect("class not defined").1.clone()
    }

    /// iterate over defaultable types
    pub fn defaults(&self) -> impl Iterator<Item = &Type> {
        self.defaults.iter()
    }

    /// test if a class is defined
    pub fn is_defined(&self, name: &Id) -> bool {
        self.classes.get(name).is_some()
    }

    /// add a new or updated class definition
    pub fn modify(&self, name: Id, cls: Class) -> Self {
        ClassEnv {
            classes: self.classes.insert(name, cls),
            defaults: self.defaults.clone(),
        }
    }

    pub fn by_super(&self, p: Pred) -> List<Pred> {
        match &p {
            Pred::IsIn(i, t) => List::concat(
                self.supers(i)
                    .iter()
                    .map(|i_| self.by_super(Pred::IsIn(i_.clone(), t.clone()))),
            )
            .cons(p),
        }
    }

    pub fn by_inst(&self, p: &Pred) -> crate::Result<List<Pred>> {
        match p {
            Pred::IsIn(i, _) => self
                .insts(i)
                .iter()
                .map(|Qual(ps, h)| {
                    let u = match_pred(h, p)?;
                    Ok(ps.iter().map(|p_| u.apply(p_)).collect())
                })
                .filter(crate::Result::is_ok)
                .map(crate::Result::unwrap)
                .next()
                .ok_or_else(|| "no matching instance".to_string()),
        }
    }

    pub fn entail(&self, ps: &[Pred], p: &Pred) -> bool {
        ps.iter()
            .cloned()
            .map(|p_| self.by_super(p_))
            .any(|sup| sup.contains(p))
            || match self.by_inst(p) {
                Err(_) => false,
                Ok(qs) => qs.iter().all(|_| self.entail(ps, p)),
            }
    }

    pub fn to_hnfs<'a>(&self, ps: impl IntoIterator<Item = &'a Pred>) -> crate::Result<Vec<Pred>> {
        let tmp: crate::Result<Vec<_>> = ps.into_iter().map(|p| self.to_hnf(&p)).collect();
        Ok(tmp?.into_iter().flatten().collect())
    }

    pub fn to_hnf(&self, p: &Pred) -> crate::Result<Vec<Pred>> {
        if p.in_hnf() {
            Ok(vec![p.clone()])
        } else {
            match self.by_inst(p) {
                Err(e) => Err(format!("context reduction ({e}): {p:?}"))?,
                Ok(ps) => self.to_hnfs(&ps),
            }
        }
    }

    pub fn simplify(&self, mut ps: Vec<Pred>) -> Vec<Pred> {
        let mut rs = vec![];

        while let Some(p) = ps.pop() {
            let mut rsps = rs.clone();
            rsps.extend(ps.clone());
            if !self.entail(&rsps, &p) {
                rs.push(p)
            }
        }

        rs
    }

    pub fn reduce(&self, ps: &[Pred]) -> crate::Result<Vec<Pred>> {
        let qs = self.to_hnfs(ps)?;
        Ok(self.simplify(qs))
    }
}

pub struct EnvTransformer(Rc<dyn Fn(&ClassEnv) -> crate::Result<ClassEnv>>);

impl EnvTransformer {
    pub fn new() -> Self {
        EnvTransformer(Rc::new(|ce| {
            Ok(ClassEnv {
                classes: ce.classes.clone(),
                defaults: ce.defaults.clone(),
            })
        }))
    }

    pub fn apply(&self, ce: &ClassEnv) -> crate::Result<ClassEnv> {
        self.0(ce)
    }

    pub fn compose(self, other: Self) -> Self {
        EnvTransformer(Rc::new(move |ce| {
            let ce_ = self.0(ce)?;
            other.0(&ce_)
        }))
    }

    pub fn add_class(i: Id, sis: List<Id>) -> Self {
        EnvTransformer(Rc::new(move |ce| {
            if ce.is_defined(&i) {
                Err("class {i} already defined")?
            }
            for j in sis.iter() {
                if !ce.is_defined(j) {
                    Err("superclass {j} not defined")?
                }
            }
            Ok(ce.modify(i.clone(), Class(sis.clone(), list![])))
        }))
    }

    pub fn add_inst(ps: Vec<Pred>, p: Pred) -> Self {
        EnvTransformer(Rc::new(move |ce| match &p {
            Pred::IsIn(i, _) => {
                if !ce.is_defined(&i) {
                    Err("no class for instance")?
                }
                let its = ce.insts(&i);
                let mut qs = its.iter().map(|Qual(_, q)| q);
                if qs.any(|q| overlap(&p, q)) {
                    Err("overlapping instance")?
                }
                let c = Class(ce.supers(i), its.cons(Qual(ps.clone(), p.clone())));
                Ok(ce.modify(i.clone(), c))
            }
        }))
    }
}

fn overlap(p: &Pred, q: &Pred) -> bool {
    mgu_pred(p, q).is_ok()
}
