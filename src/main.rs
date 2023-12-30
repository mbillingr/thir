// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

#[macro_export]
macro_rules! list {
    () => { List::Nil };

    ($x:expr $(, $r:expr)*) => {
        list![$($r),*].cons($x)
    };
}

mod assumptions;
mod classes;
mod instantiate;
mod kinds;
mod predicates;
mod qualified;
mod scheme;
mod specific_inference;
mod specifics;
mod substitutions;
mod type_inference;
mod types;
mod unification;

use crate::assumptions::Assump;
use crate::classes::{ClassEnv, EnvTransformer};
use crate::kinds::{HasKind, Kind};
use crate::predicates::{match_pred, mgu_pred};
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{
    ti_program, Alt, BindGroup, Expl, Expr, Impl, Literal, Pat, Program,
};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::substitutions::{Subst, Types};
use crate::type_inference::TI;
use crate::types::{Type, Tyvar};
use crate::unification::{matches, mgu};
use predicates::{Pred, Pred::IsIn};
use std::fmt::{Debug, Formatter};
use std::iter::once;
use std::rc::Rc;

type Result<T> = std::result::Result<T, String>;

fn main() {
    let ce = ClassEnv::default();
    let ce = add_core_classes().apply(&ce).unwrap();
    let ce = add_num_classes().apply(&ce).unwrap();

    let prog = Program(vec![BindGroup(
        vec![
            /*Expl(
                "foo".into(),
                Scheme::Forall(List::Nil, Qual(vec![], Type::t_int())),
                vec![Alt(vec![], Expr::Var("bar".into()))],
            ),*/
            /*Expl(
                "ident".into(),
                Scheme::Forall(
                    list![Kind::Star],
                    Qual(vec![], Type::func(Type::TGen(0), Type::TGen(0))),
                ),
                vec![Alt(vec![Pat::PVar("x".into())], Expr::Var("x".into()))],
            ),*/
            /*Expl(
                "a-const".into(),
                Scheme::Forall(list![], Qual(vec![], Type::t_int())),
                vec![Alt(vec![], Expr::Lit(Literal::Int(42)).into())],
            ),*/
        ],
        vec![vec![
            // todo: why do the implicits (in particular, the constant) result in generic types?
            Impl(
                "a-const".into(),
                vec![Alt(vec![], Expr::Lit(Literal::Int(42)).into())],
            ),
            /*Impl(
                "bar".into(),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("ident".into()).into(),
                        Expr::Lit(Literal::Int(42)).into(),
                    ),
                )],
            ),
            Impl(
                "baz".into(),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("ident".into()).into(),
                        Expr::Var("ident".into()).into(),
                    ),
                )],
            ),*/
        ]],
    )]);

    let r = ti_program(&ce, vec![], &prog);
    println!("{r:#?}")
}

type Int = usize;
type Id = String;

fn enum_id(n: Int) -> Id {
    format!("v{n}")
}

fn eq_union<T: PartialEq>(mut a: Vec<T>, b: Vec<T>) -> Vec<T> {
    for x in b {
        if !a.contains(&x) {
            a.push(x)
        }
    }
    a
}

fn eq_intersect<T: PartialEq>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    let mut out = vec![];
    for x in b {
        if a.contains(&x) {
            out.push(x)
        }
    }
    out
}

pub fn quantify(vs: &[Tyvar], qt: &Qual<Type>) -> Scheme {
    let vs_: Vec<_> = qt.tv().into_iter().filter(|v| vs.contains(v)).collect();
    let ks = vs_.iter().map(|v| v.kind().unwrap().clone()).collect();
    let n = vs_.len() as Int;
    let s = Subst::from_rev_iter(
        vs_.into_iter()
            .rev()
            .zip((0..n).rev().map(|k| Type::TGen(k))),
    );
    Scheme::Forall(ks, s.apply(qt))
}

pub fn to_scheme(t: Type) -> Scheme {
    Scheme::Forall(List::Nil, Qual(vec![], t))
}

fn find<'a>(i: &Id, ass: impl IntoIterator<Item = &'a Assump>) -> Result<&'a Scheme> {
    for a in ass {
        if &a.i == i {
            return Ok(&a.sc);
        }
    }
    Err(format!("unbound identifier: {i}"))
}

struct Ambiguity(Tyvar, Vec<Pred>);

fn ambiguities(ce: &ClassEnv, vs: Vec<Tyvar>, ps: &[Pred]) -> Vec<Ambiguity> {
    let mut out = vec![];
    for v in list_diff(ps.tv(), vs) {
        let ps_ = ps.iter().filter(|p| p.tv().contains(&v)).cloned().collect();
        out.push(Ambiguity(v, ps_))
    }
    out
}

const NUM_CLASSES: [&str; 7] = [
    "Num",
    "Integral",
    "Floating",
    "Fractional",
    "Real",
    "RealFloat",
    "RealFrac",
];
const STD_CLASSES: [&str; 17] = [
    "Eq",
    "Ord",
    "Show",
    "Read",
    "Bounded",
    "Enum",
    "Ix",
    "Functor",
    "Monad",
    "MonadPlus",
    "Num",
    "Integral",
    "Floating",
    "Fractional",
    "Real",
    "RealFloat",
    "RealFrac",
];

fn candidates(ce: &ClassEnv, Ambiguity(v, qs): &Ambiguity) -> Vec<Type> {
    let is_ = || qs.iter().map(|Pred::IsIn(i, t)| i);
    let ts_: Vec<_> = qs.iter().map(|Pred::IsIn(i, t)| t).collect();

    if !ts_
        .into_iter()
        .all(|t| if let Type::TVar(u) = t { u == v } else { false })
    {
        return vec![];
    }

    if !is_().any(|i| NUM_CLASSES.contains(&i.as_str())) {
        return vec![];
    }

    if !is_().all(|i| STD_CLASSES.contains(&i.as_str())) {
        return vec![];
    }

    let mut out = vec![];
    for t_ in ce.defaults() {
        if is_()
            .map(|i| Pred::IsIn(i.clone(), t_.clone()))
            .all(|p| ce.entail(&[], &p))
        {
            out.push(t_.clone());
        }
    }

    out
}

fn with_defaults<T>(
    f: impl Fn(Vec<Ambiguity>, Vec<Type>) -> T,
    ce: &ClassEnv,
    vs: Vec<Tyvar>,
    ps: &[Pred],
) -> Result<T> {
    let vps = ambiguities(ce, vs, ps);
    let tss = vps.iter().map(|vp| candidates(ce, vp));

    let mut heads = Vec::with_capacity(vps.len());
    for ts in tss {
        heads.push(
            ts.into_iter()
                .next()
                .ok_or_else(|| "cannot resolve ambiguity")?,
        )
    }

    Ok(f(vps, heads))
}

fn defaulted_preds(ce: &ClassEnv, vs: Vec<Tyvar>, ps: &[Pred]) -> Result<Vec<Pred>> {
    with_defaults(
        |vps, ts| vps.into_iter().map(|Ambiguity(_, p)| p).flatten().collect(),
        ce,
        vs,
        ps,
    )
}

fn default_subst(ce: &ClassEnv, vs: Vec<Tyvar>, ps: &[Pred]) -> Result<Subst> {
    with_defaults(
        |vps, ts| Subst::from_rev_iter(vps.into_iter().map(|Ambiguity(v, _)| v).zip(ts).rev()),
        ce,
        vs,
        ps,
    )
}

// ============================================

fn rfold1<T, I: DoubleEndedIterator<Item = T>>(
    it: impl IntoIterator<Item = T, IntoIter = I>,
    f: impl Fn(T, T) -> T,
) -> T {
    let mut it = it.into_iter().rev();
    let mut res = it.next().expect("List with at least one element");
    while let Some(x) = it.next() {
        res = f(res, x);
    }
    res
}

fn list_diff<T: PartialEq>(a: impl IntoIterator<Item = T>, mut b: Vec<T>) -> Vec<T> {
    let mut out = vec![];
    for x in a {
        if let Some(i) = b.iter().position(|y| &x == y) {
            b.swap_remove(i);
        } else {
            out.push(x);
        }
    }
    out
}

fn list_union<T: PartialEq>(mut a: Vec<T>, b: impl IntoIterator<Item = T>) -> Vec<T> {
    for x in b {
        if !a.contains(&x) {
            a.push(x)
        }
    }
    a
}

fn list_intersect<T: PartialEq>(a: impl IntoIterator<Item = T>, mut b: Vec<T>) -> Vec<T> {
    a.into_iter().filter(|x| b.contains(x)).collect()
}

#[derive(PartialEq)]
enum List<T> {
    Nil,
    Elem(Rc<(T, Self)>),
}

impl<T: Debug> Debug for List<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut es = self.iter();
        if let Some(e) = es.next() {
            write!(f, "{:?}", e)?;
        }
        for e in es {
            write!(f, " {:?}", e)?;
        }
        write!(f, "]")
    }
}

impl<T> List<T> {
    fn cons(&self, x: T) -> Self {
        List::Elem(Rc::new((x, self.clone())))
    }

    fn iter(&self) -> ListIter<T> {
        ListIter(&self)
    }

    fn concat<I>(ls: I) -> Self
    where
        T: Clone,
        I: IntoIterator<Item = Self>,
        I::IntoIter: DoubleEndedIterator,
    {
        let mut out = Self::Nil;
        for l in ls.into_iter().rev() {
            out = l.append(out)
        }
        out
    }

    fn append(&self, b: Self) -> Self
    where
        T: Clone,
    {
        match self {
            Self::Nil => b,
            Self::Elem(e) => e.1.append(b).cons(e.0.clone()),
        }
    }

    fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        match self {
            Self::Nil => false,
            Self::Elem(e) if &e.0 == x => true,
            Self::Elem(e) => e.1.contains(x),
        }
    }
}

impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        match self {
            List::Nil => List::Nil,
            List::Elem(e) => List::Elem(e.clone()),
        }
    }
}

struct ListIter<'a, T>(&'a List<T>);

impl<'a, T> Iterator for ListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let e = match &self.0 {
            List::Nil => return None,
            List::Elem(e) => e,
        };

        self.0 = &e.1;
        return Some(&e.0);
    }
}

impl<T> FromIterator<T> for List<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut items: Vec<_> = iter.into_iter().collect();
        let mut out = List::Nil;
        while let Some(x) = items.pop() {
            out = out.cons(x);
        }
        out
    }
}

impl<'a, T> IntoIterator for &'a List<T> {
    type Item = &'a T;
    type IntoIter = ListIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        ListIter(self)
    }
}
