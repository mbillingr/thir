// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

mod kinds;
mod types;
mod types_specific;

use crate::kinds::Kind;
use crate::types::{Tycon, Type, Tyvar};
use crate::Pred::IsIn;
use std::iter::once;
use std::rc::Rc;

type Result<T> = std::result::Result<T, String>;

macro_rules! list {
    () => { List::Nil };

    ($($x:expr),*) => {{
        let mut lst = List::Nil;
        $(
            lst = lst.cons($x);
        )*
        lst
    }};
}

fn main() {
    let ce = ClassEnv::default();
    let ce = add_core_classes()(&ce).unwrap();
    let ce = add_num_classes()(&ce).unwrap();

    let prog = Program(vec![BindGroup(
        vec![
            /*Expl(
                "foo".into(),
                Scheme::Forall(List::Nil, Qual(vec![], Type::t_int())),
                vec![Alt(vec![], Expr::Var("bar".into()))],
            ),
            Expl(
                "ident".into(),
                Scheme::Forall(
                    list![Kind::Star],
                    Qual(vec![], Type::func(Type::TGen(0), Type::TGen(0))),
                ),
                vec![Alt(vec![Pat::PVar("x".into())], Expr::Var("x".into()))],
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
    println!("{r:?}")
}

type Int = usize;
type Id = String;

fn enum_id(n: Int) -> Id {
    format!("v{n}")
}

#[derive(Clone, Debug)]
enum Subst {
    Empty,
    Assoc(Rc<(Tyvar, Type, Subst)>),
}

impl Subst {
    pub fn null_subst() -> Self {
        Self::Empty
    }

    pub fn single(u: Tyvar, t: Type) -> Self {
        Self::Assoc(Rc::new((u, t, Self::Empty)))
    }

    pub fn from_rev_iter(it: impl IntoIterator<Item = (Tyvar, Type)>) -> Self {
        let mut out = Self::Empty;

        for (u, t) in it {
            out = Self::Assoc(Rc::new((u, t, out)));
        }

        out
    }

    pub fn lookup(&self, u: &Tyvar) -> Option<&Type> {
        match self {
            Self::Empty => None,
            Self::Assoc(ass) if &ass.0 == u => Some(&ass.1),
            Self::Assoc(ass) => ass.2.lookup(u),
        }
    }

    pub fn keys(&self) -> Vec<Tyvar> {
        let mut out = vec![];

        let mut cursor = self;
        while let Subst::Assoc(ass) = cursor {
            let (u, t, nxt) = &**ass;
            cursor = nxt;
            out.push(u.clone());
        }

        out
    }

    /// @@ operator
    pub fn compose(&self, other: &Self) -> Self {
        let mut out = self.clone();

        let mut cursor = other;
        while let Subst::Assoc(ass) = cursor {
            let (u, t, nxt) = &**ass;
            cursor = nxt;
            out = Self::Assoc(Rc::new((u.clone(), t.clone(), out)));
        }

        out
    }

    pub fn merge(&self, other: &Self) -> Result<Self> {
        for v in eq_intersect(self.keys(), other.keys()) {
            if apply(self, &Type::TVar(v.clone())) != apply(other, &Type::TVar(v)) {
                Err("merge fails")?
            }
        }

        let mut out = other.clone();

        let mut cursor = self;
        while let Subst::Assoc(ass) = cursor {
            let (u, t, nxt) = &**ass;
            cursor = nxt;
            out = Self::Assoc(Rc::new((u.clone(), t.clone(), out)));
        }

        Ok(out)
    }
}

trait Types<T: ?Sized = Self> {
    fn apply_subst(&self, s: &Subst) -> T;
    fn tv(&self) -> Vec<Tyvar>;
}

fn apply<U, T: Types<U> + ?Sized>(s: &Subst, this: &T) -> U {
    this.apply_subst(s)
}

impl Types for Type {
    fn apply_subst(&self, s: &Subst) -> Self {
        match self {
            Type::TVar(u) => match s.lookup(u) {
                Some(t) => t.clone(),
                None => Type::TVar(u.clone()),
            },
            Type::TApp(app) => Type::tapp(apply(s, &app.0), apply(s, &app.1)),
            _ => self.clone(),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Type::TVar(u) => vec![u.clone()],
            Type::TApp(app) => eq_union(app.0.tv(), app.1.tv()),
            _ => vec![],
        }
    }
}

impl<T: Types> Types for Vec<T> {
    fn apply_subst(&self, s: &Subst) -> Self {
        self.iter().map(|x| apply(s, x)).collect()
    }

    fn tv(&self) -> Vec<Tyvar> {
        let mut tvs = vec![];
        for x in self {
            for u in x.tv() {
                if !tvs.contains(&u) {
                    tvs.push(u)
                }
            }
        }
        tvs
    }
}

impl<T: Types> Types<Vec<T>> for [T] {
    fn apply_subst(&self, s: &Subst) -> Vec<T> {
        self.iter().map(|x| apply(s, x)).collect()
    }

    fn tv(&self) -> Vec<Tyvar> {
        let mut tvs = vec![];
        for x in self {
            for u in x.tv() {
                if !tvs.contains(&u) {
                    tvs.push(u)
                }
            }
        }
        tvs
    }
}

impl<T: Types> Types for List<T> {
    fn apply_subst(&self, s: &Subst) -> Self {
        self.iter().map(|x| apply(s, x)).collect()
    }

    fn tv(&self) -> Vec<Tyvar> {
        let mut tvs = vec![];
        for x in self {
            for u in x.tv() {
                if !tvs.contains(&u) {
                    tvs.push(u)
                }
            }
        }
        tvs
    }
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

fn mgu(a: &Type, b: &Type) -> Result<Subst> {
    use Type::*;
    match (a, b) {
        (TApp(app1), TApp(app2)) => {
            let s1 = mgu(&app1.0, &app2.0)?;
            let s2 = mgu(&apply(&s1, &app1.1), &apply(&s1, &app2.1))?;
            Ok(s2.compose(&s1))
        }

        (TVar(u), t) => var_bind(u, t),

        (t, TVar(u)) => var_bind(u, t),

        (TCon(tc1), TCon(tc2)) if tc1 == tc2 => Ok(Subst::Empty),

        _ => Err(format!("types do not unify: {a:?}, {b:?}"))?,
    }
}

fn var_bind(u: &Tyvar, t: &Type) -> Result<Subst> {
    if let Type::TVar(v) = t {
        if u == v {
            return Ok(Subst::Empty);
        }
    }

    if t.tv().contains(u) {
        Err("occurs check failed")?
    }

    if u.kind() != t.kind() {
        Err("kinds do not match")?
    }

    Ok(Subst::single(u.clone(), t.clone()))
}

fn matches(a: &Type, b: &Type) -> Result<Subst> {
    use Type::*;
    match (a, b) {
        (TApp(app1), TApp(app2)) => {
            let sl = matches(&app1.0, &app2.0)?;
            let sr = matches(&app1.1, &app2.1)?;
            sl.merge(&sr)
        }

        (TVar(u), t) if u.kind() == t.kind() => Ok(Subst::single(u.clone(), t.clone())),

        (TCon(tc1), TCon(tc2)) if tc1 == tc2 => Ok(Subst::Empty),

        _ => Err("types do not match")?,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Qual<T>(Vec<Pred>, T);

#[derive(Clone, Debug, PartialEq)]
enum Pred {
    IsIn(Id, Type),
}

impl<T: Types> Types for Qual<T> {
    fn apply_subst(&self, s: &Subst) -> Self {
        Qual(apply(s, &self.0), apply(s, &self.1))
    }

    fn tv(&self) -> Vec<Tyvar> {
        eq_union(self.0.tv(), self.1.tv())
    }
}

impl Types for Pred {
    fn apply_subst(&self, s: &Subst) -> Self {
        match self {
            Pred::IsIn(i, t) => Pred::IsIn(i.clone(), apply(s, t)),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Pred::IsIn(i, t) => t.tv(),
        }
    }
}

fn mgu_pred(a: &Pred, b: &Pred) -> Result<Subst> {
    lift(mgu, a, b)
}

fn match_pred(a: &Pred, b: &Pred) -> Result<Subst> {
    lift(matches, a, b)
}

fn lift(m: impl Fn(&Type, &Type) -> Result<Subst>, a: &Pred, b: &Pred) -> Result<Subst> {
    match (a, b) {
        (Pred::IsIn(i1, t1), Pred::IsIn(i2, t2)) if i1 == i2 => m(t1, t2),
        _ => Err("classes differ")?,
    }
}

#[derive(Debug, Clone)]
struct Class(Rc<Vec<Id>>, List<Inst>);

type Inst = Qual<Pred>;

struct ClassEnv {
    classes: Rc<dyn Fn(&Id) -> Result<Class>>,
    defaults: List<Type>,
}

impl Default for ClassEnv {
    fn default() -> Self {
        ClassEnv {
            classes: Rc::new(|i| Err(format!("class {i} not defined"))?),
            defaults: list![Type::t_int(), Type::t_double()],
        }
    }
}

impl ClassEnv {
    pub fn supers(&self, name: &Id) -> Rc<Vec<Id>> {
        (self.classes)(name).unwrap().0
    }

    pub fn insts(&self, name: &Id) -> List<Inst> {
        (self.classes)(name).unwrap().1
    }

    pub fn is_defined(&self, name: &Id) -> bool {
        (self.classes)(name).is_ok()
    }

    pub fn modify(&self, name: Id, cls: Class) -> Self {
        let next = self.classes.clone();
        ClassEnv {
            classes: Rc::new(move |j| if j == &name { Ok(cls.clone()) } else { next(j) }),
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

    pub fn by_inst(&self, p: &Pred) -> Result<List<Pred>> {
        match p {
            Pred::IsIn(i, t) => self
                .insts(i)
                .iter()
                .map(|Qual(ps, h)| {
                    let u = match_pred(h, p)?;
                    Ok(ps.iter().map(|p_| apply(&u, p_)).collect())
                })
                .filter(Result::is_ok)
                .map(Result::unwrap)
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
                Ok(qs) => qs.iter().all(|q| self.entail(ps, p)),
            }
    }

    pub fn to_hnfs<'a>(&self, ps: impl IntoIterator<Item = &'a Pred>) -> Result<Vec<Pred>> {
        let tmp: Result<Vec<_>> = ps.into_iter().map(|p| self.to_hnf(&p)).collect();
        Ok(tmp?.into_iter().flatten().collect())
    }

    pub fn to_hnf(&self, p: &Pred) -> Result<Vec<Pred>> {
        if in_hnf(p) {
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

    pub fn reduce(&self, ps: &[Pred]) -> Result<Vec<Pred>> {
        let qs = self.to_hnfs(ps)?;
        Ok(self.simplify(qs))
    }
}

/// test if a predicate is in head-normal form
pub fn in_hnf(p: &Pred) -> bool {
    fn hnf(t: &Type) -> bool {
        match t {
            Type::TVar(_) => true,
            Type::TCon(_) => false,
            Type::TApp(app) => hnf(&app.0),
            Type::TGen(_) => panic!("don't know what to do!"),
        }
    }

    match p {
        Pred::IsIn(_, t) => hnf(t),
    }
}

type EnvTransformer = Rc<dyn Fn(&ClassEnv) -> Result<ClassEnv>>;

fn compose_transformers(f: EnvTransformer, g: EnvTransformer) -> EnvTransformer {
    Rc::new(move |ce| {
        let ce_ = f(ce)?;
        g(&ce_)
    })
}

fn add_class(i: Id, sis: Vec<Id>) -> EnvTransformer {
    let sis = Rc::new(sis);
    Rc::new(move |ce| {
        if ce.is_defined(&i) {
            Err("class {i} already defined")?
        }
        for j in sis.iter() {
            if !ce.is_defined(j) {
                Err("superclass {j} not defined")?
            }
        }
        Ok(ce.modify(i.clone(), Class(sis.clone(), list![])))
    })
}

fn add_inst(ps: Vec<Pred>, p: Pred) -> EnvTransformer {
    Rc::new(move |ce| match &p {
        Pred::IsIn(i, _) => {
            let its = ce.insts(&i);
            let mut qs = its.iter().map(|Qual(_, q)| q);
            let c = Class(ce.supers(i), its.cons(Qual(ps.clone(), p.clone())));
            if !ce.is_defined(&i) {
                Err("no class for instance")?
            }
            if qs.any(|q| overlap(&p, q)) {
                Err("overlapping instance")?
            }
            Ok(ce.modify(i.clone(), c))
        }
    })
}

fn add_core_classes() -> EnvTransformer {
    let et = add_class("Eq".into(), vec![]);
    let et = compose_transformers(et, add_class("Ord".into(), vec!["Eq".into()]));
    let et = compose_transformers(et, add_class("Show".into(), vec![]));
    let et = compose_transformers(et, add_class("Read".into(), vec![]));
    let et = compose_transformers(et, add_class("Bounded".into(), vec![]));
    let et = compose_transformers(et, add_class("Enum".into(), vec![]));
    let et = compose_transformers(et, add_class("Functor".into(), vec![]));
    let et = compose_transformers(et, add_class("Monad".into(), vec![]));
    et
}

fn add_num_classes() -> EnvTransformer {
    let et = add_class("Num".into(), vec!["Eq".into(), "Show".into()]);
    let et = compose_transformers(
        et,
        add_class("Real".into(), vec!["Num".into(), "Ord".into()]),
    );
    let et = compose_transformers(et, add_class("Fractional".into(), vec!["Num".into()]));
    let et = compose_transformers(
        et,
        add_class("Integral".into(), vec!["Real".into(), "Enum".into()]),
    );
    let et = compose_transformers(
        et,
        add_class("RealFrac".into(), vec!["Real".into(), "Fractional".into()]),
    );
    let et = compose_transformers(et, add_class("Floating".into(), vec!["Fractional".into()]));
    let et = compose_transformers(
        et,
        add_class(
            "RealFloat".into(),
            vec!["RealFrac".into(), "Floating".into()],
        ),
    );

    let et = compose_transformers(et, add_inst(vec![], IsIn("Num".into(), Type::t_int())));

    et
}

fn overlap(p: &Pred, q: &Pred) -> bool {
    mgu_pred(p, q).is_ok()
}

#[derive(Clone, Debug, PartialEq)]
enum Scheme {
    Forall(List<Kind>, Qual<Type>),
}

impl Types for Scheme {
    fn apply_subst(&self, s: &Subst) -> Self {
        match self {
            Scheme::Forall(ks, qt) => Scheme::Forall(ks.clone(), apply(s, qt)),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Scheme::Forall(_, qt) => qt.tv(),
        }
    }
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
    Scheme::Forall(ks, apply(&s, qt))
}

pub fn to_scheme(t: Type) -> Scheme {
    Scheme::Forall(List::Nil, Qual(vec![], t))
}

#[derive(Clone, Debug)]
struct Assump {
    i: Id,
    sc: Scheme,
}

impl Types for Assump {
    fn apply_subst(&self, s: &Subst) -> Self {
        Assump {
            i: self.i.clone(),
            sc: apply(s, &self.sc),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        self.sc.tv()
    }
}

fn find<'a>(i: &Id, ass: impl IntoIterator<Item = &'a Assump>) -> Result<&'a Scheme> {
    for a in ass {
        if &a.i == i {
            return Ok(&a.sc);
        }
    }
    Err(format!("unbound identifier: {i}"))
}

/// The type inference state
struct TI {
    subst: Subst,
    count: Int,
}

impl TI {
    pub fn new() -> Self {
        TI {
            subst: Subst::null_subst(),
            count: 0,
        }
    }

    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<()> {
        let u = mgu(&apply(&self.subst, t1), &apply(&self.subst, t2))?;
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
                inst(&ts, qt)
            }
        }
    }

    fn ext_subst(&mut self, s: Subst) {
        self.subst = s.compose(&self.subst); // todo: is the order of composition correct?
    }
}

fn inst<T: Instantiate>(ts: &[Type], t: &T) -> T {
    t.inst(ts)
}

trait Instantiate {
    fn inst(&self, ts: &[Type]) -> Self;
}

impl Instantiate for Type {
    fn inst(&self, ts: &[Type]) -> Self {
        match self {
            Type::TApp(app) => Type::tapp(app.0.inst(ts), app.1.inst(ts)),
            Type::TGen(n) => ts[*n].clone(),
            t => t.clone(),
        }
    }
}

impl<T: Instantiate> Instantiate for Vec<T> {
    fn inst(&self, ts: &[Type]) -> Self {
        self.iter().map(|t| inst(ts, t)).collect()
    }
}

impl<T: Instantiate> Instantiate for List<T> {
    fn inst(&self, ts: &[Type]) -> Self {
        self.iter().map(|t| inst(ts, t)).collect()
    }
}

impl<T: Instantiate> Instantiate for Qual<T> {
    fn inst(&self, ts: &[Type]) -> Self {
        Qual(inst(ts, &self.0), inst(ts, &self.1))
    }
}

impl Instantiate for Pred {
    fn inst(&self, ts: &[Type]) -> Self {
        match self {
            Pred::IsIn(c, t) => Pred::IsIn(c.clone(), inst(ts, t)),
        }
    }
}

enum Literal {
    Int(i64),
    Char(char),
    Rat(f64),
    Str(String),
}

fn ti_lit(ti: &mut TI, l: &Literal) -> Result<(Vec<Pred>, Type)> {
    match l {
        Literal::Char(_) => Ok((vec![], Type::t_char())),
        Literal::Str(_) => Ok((vec![], Type::t_string())),
        Literal::Int(_) => {
            let v = ti.new_tvar(Kind::Star);
            Ok((vec![Pred::IsIn("Num".into(), v.clone())], v))
        }
        Literal::Rat(_) => {
            let v = ti.new_tvar(Kind::Star);
            Ok((vec![Pred::IsIn("Fractional".into(), v.clone())], v))
        }
    }
}

enum Pat {
    PVar(Id),
    PWildcard,
    PAs(Id, Rc<Pat>),
    PLit(Literal),
    PNpk(Id, Int),
    PCon(Assump, Vec<Pat>),
}

fn ti_pat(ti: &mut TI, pat: &Pat) -> Result<(Vec<Pred>, Vec<Assump>, Type)> {
    match pat {
        Pat::PVar(i) => {
            let v = ti.new_tvar(Kind::Star);
            Ok((
                vec![],
                vec![Assump {
                    i: i.clone(),
                    sc: to_scheme(v.clone()),
                }],
                v,
            ))
        }

        Pat::PWildcard => {
            let v = ti.new_tvar(Kind::Star);
            Ok((vec![], vec![], v))
        }

        Pat::PAs(i, p) => {
            let (ps, mut as_, t) = ti_pat(ti, p)?;

            // todo: does order of assumptions matter? the paper conses it to the front.
            as_.push(Assump {
                i: i.clone(),
                sc: to_scheme(t.clone()),
            });

            Ok((ps, as_, t))
        }

        Pat::PLit(li) => {
            let (ps, t) = ti_lit(ti, li)?;
            Ok((ps, vec![], t))
        }

        Pat::PNpk(i, k) => {
            let t = ti.new_tvar(Kind::Star);
            Ok((
                vec![Pred::IsIn("Integral".into(), t.clone())],
                vec![Assump {
                    i: i.clone(),
                    sc: to_scheme(t.clone()),
                }],
                t,
            ))
        }

        Pat::PCon(Assump { i, sc }, pats) => {
            let (mut ps, as_, ts) = ti_pats(ti, pats)?;
            let t_ = ti.new_tvar(Kind::Star);
            let Qual(qs, t) = ti.fresh_inst(sc);
            let f = ts.into_iter().rfold(t_.clone(), Type::func);
            ti.unify(&t, &f)?;
            ps.extend(qs);
            Ok((ps, as_, t_))
        }
    }
}

fn ti_pats(ti: &mut TI, pats: &[Pat]) -> Result<(Vec<Pred>, Vec<Assump>, Vec<Type>)> {
    let psats = pats
        .iter()
        .map(|p| ti_pat(ti, p))
        .collect::<Result<Vec<_>>>()?;

    let mut ps = vec![];
    let mut as_ = vec![];
    let mut ts = vec![];

    for (ps_, as__, t) in psats {
        ps.extend(ps_);
        as_.extend(as__);
        ts.push(t);
    }

    Ok((ps, as_, ts))
}

enum Expr {
    Var(Id),
    Lit(Literal),
    Const(Assump),
    App(Rc<Expr>, Rc<Expr>),
    Let(BindGroup, Rc<Expr>),
}

fn ti_expr(ti: &mut TI, ce: &ClassEnv, ass: &[Assump], expr: &Expr) -> Result<(Vec<Pred>, Type)> {
    match expr {
        Expr::Var(i) => {
            let sc = find(i, ass)?;
            let Qual(ps, t) = ti.fresh_inst(sc);
            Ok((ps, t))
        }

        Expr::Const(Assump { i, sc }) => {
            let Qual(ps, t) = ti.fresh_inst(sc);
            Ok((ps, t))
        }

        Expr::Lit(li) => ti_lit(ti, li),

        Expr::App(e, f) => {
            let (mut ps, te) = ti_expr(ti, ce, ass, e)?;
            let (qs, tf) = ti_expr(ti, ce, ass, f)?;
            let t = ti.new_tvar(Kind::Star);
            ti.unify(&Type::func(tf, t.clone()), &te)?;
            ps.extend(qs);
            Ok((ps, t))
        }

        Expr::Let(bg, e) => {
            let (mut ps, mut ass_) = ti_bindgroup(ti, ce, ass, bg)?;
            ass_.extend(ass.iter().cloned());
            let (qs, t) = ti_expr(ti, ce, &ass_, e)?;
            ps.extend(qs);
            Ok((ps, t))
        }
    }
}

struct Alt(Vec<Pat>, Expr);

fn ti_alt(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    Alt(pats, e): &Alt,
) -> Result<(Vec<Pred>, Type)> {
    let (mut ps, mut ass_, ts) = ti_pats(ti, pats)?;
    ass_.extend(ass.iter().cloned());
    let (qs, t) = ti_expr(ti, ce, &ass_, e)?;
    ps.extend(qs);
    let f = ts.into_iter().rfold(t, Type::func);
    Ok((ps, f))
}

fn ti_alts(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    alts: &[Alt],
    t: &Type,
) -> Result<Vec<Pred>> {
    let psts = alts
        .iter()
        .map(|a| ti_alt(ti, ce, ass, a))
        .collect::<Result<Vec<_>>>()?;
    let mut ps = vec![];
    for (ps_, t_) in psts {
        ti.unify(t, &t_)?;
        ps.extend(ps_);
    }
    Ok(ps)
}

fn split(ce: &ClassEnv, fs: &[Tyvar], gs: &[Tyvar], ps: &[Pred]) -> Result<(Vec<Pred>, Vec<Pred>)> {
    let ps_ = ce.reduce(ps)?;
    let (ds, rs): (Vec<_>, _) = ps_
        .into_iter()
        .partition(|p| p.tv().iter().all(|tv| fs.contains(tv)));
    let mut fsgs = vec![];
    fsgs.extend(fs.iter().chain(gs.iter()).cloned());
    let rs_ = defaulted_preds(ce, fsgs, &rs)?;
    Ok((ds, list_diff(rs, rs_)))
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
    for t_ in &ce.defaults {
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

/// Explicitly typed binding
struct Expl(Id, Scheme, Vec<Alt>);

fn ti_expl(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    Expl(i, sc, alts): &Expl,
) -> Result<(Vec<Pred>)> {
    let Qual(qs, t) = ti.fresh_inst(sc);
    let ps = ti_alts(ti, ce, ass, alts, &t)?;
    let s = &ti.subst;
    let qs_ = apply(s, &qs);
    let t_ = apply(s, &t);
    let fs = apply(s, ass).tv();
    let gs = list_diff(t_.tv(), fs.clone());
    let ps_: Vec<_> = apply(s, &ps)
        .into_iter()
        .filter(|p| !ce.entail(&qs_, p))
        .collect();
    let sc_ = quantify(&gs, &Qual(qs_, t_));
    let (ds, rs) = split(ce, &fs, &gs, &ps_)?;

    if sc != &sc_ {
        Err(format!("signature too general {sc:?} != {sc_:?}"))?;
    }

    if !rs.is_empty() {
        Err("context too weak")?;
    }

    Ok(ds)
}

/// Implicitly typed binding
struct Impl(Id, Vec<Alt>);

fn restricted(bs: &[Impl]) -> bool {
    fn simple(Impl(i, alts): &Impl) -> bool {
        alts.iter().any(|Alt(pat, _)| pat.is_empty())
    }

    bs.iter().any(simple)
}

/// Infer group of mutually recursive implicitly typed bindings
fn ti_impls(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    bs: &Vec<Impl>,
) -> Result<(Vec<Pred>, Vec<Assump>)> {
    let ts: Vec<_> = bs.iter().map(|_| ti.new_tvar(Kind::Star)).collect();
    let is = || bs.iter().map(|Impl(i, _)| i.clone());
    let scs = ts.iter().cloned().map(to_scheme);
    let as_: Vec<_> = is()
        .zip(scs)
        .map(|(i, sc)| Assump { i, sc })
        .chain(ass.iter().cloned())
        .collect();
    let altss = bs.iter().map(|Impl(_, alts)| alts);
    let pss = altss
        .zip(&ts)
        .map(|(a, t)| ti_alts(ti, ce, &as_, a, t))
        .collect::<Result<Vec<_>>>()?;
    let s = &ti.subst;
    let ps_ = apply(s, &pss.into_iter().flatten().collect::<Vec<_>>());
    let ts_ = apply(s, &ts);
    let fs = apply(s, ass).tv();
    let vss = || ts_.iter().map(Types::tv);
    let (mut ds, rs) = split(ce, &fs, &vss().rfold(vec![], list_intersect), &ps_)?;
    let gs = list_diff(vss().rfold(vec![], list_union), fs);
    if restricted(bs) {
        let gs_ = list_diff(gs, rs.tv());
        let scs_ = ts_.into_iter().map(|t| quantify(&gs_, &Qual(vec![], t)));
        ds.extend(rs);
        Ok((ds, is().zip(scs_).map(|(i, sc)| Assump { i, sc }).collect()))
    } else {
        let scs_ = ts_.into_iter().map(|t| quantify(&gs, &Qual(rs.clone(), t)));
        Ok((ds, is().zip(scs_).map(|(i, sc)| Assump { i, sc }).collect()))
    }
}

struct BindGroup(Vec<Expl>, Vec<Vec<Impl>>);

fn ti_bindgroup(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    BindGroup(es, iss): &BindGroup,
) -> Result<(Vec<Pred>, Vec<Assump>)> {
    let as_: Vec<_> = es
        .iter()
        .map(|Expl(v, sc, alts)| Assump {
            i: v.clone(),
            sc: sc.clone(),
        })
        .collect();

    let mut as_as = as_.clone();
    as_as.extend(ass.to_vec());

    let (ps, as__) = ti_seq(ti_impls, ti, ce, as_as, iss)?;

    let mut as__as_ = as__.clone();
    as__as_.extend(as_);

    let mut as__as_as = as__as_.clone();
    as__as_as.extend(ass.to_vec());

    let qss = es
        .iter()
        .map(|e| ti_expl(ti, ce, &as__as_as, e))
        .collect::<Result<Vec<_>>>()?;

    Ok((once(ps).chain(qss).flatten().collect(), as__as_))
}

fn ti_seq<T>(
    inf: impl Fn(&mut TI, &ClassEnv, &[Assump], &T) -> Result<(Vec<Pred>, Vec<Assump>)>,
    ti: &mut TI,
    ce: &ClassEnv,
    ass: Vec<Assump>,
    bss: &[T],
) -> Result<(Vec<Pred>, Vec<Assump>)> {
    if bss.is_empty() {
        return Ok((vec![], vec![]));
    }

    let bs = &bss[0];
    let bss = &bss[1..];

    let (mut ps, as_) = inf(ti, ce, &ass, bs)?;

    let mut as_as = as_.clone();
    as_as.extend(ass);

    let (qs, mut as__) = ti_seq(inf, ti, ce, as_as, bss)?;

    ps.extend(qs);
    as__.extend(as_);
    Ok((ps, as__))
}

struct Program(Vec<BindGroup>);

fn ti_program(ce: &ClassEnv, ass: Vec<Assump>, Program(bgs): &Program) -> Result<Vec<Assump>> {
    let mut ti = TI::new();
    let (ps, as_) = ti_seq(ti_bindgroup, &mut ti, ce, ass, bgs)?;
    let s = &ti.subst;
    let rs = ce.reduce(&apply(s, &ps))?;
    let s_ = default_subst(ce, vec![], &rs)?;
    Ok(apply(&s_.compose(s), &as_))
}

// ============================================

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

#[derive(Debug, PartialEq)]
enum List<T> {
    Nil,
    Elem(Rc<(T, Self)>),
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
