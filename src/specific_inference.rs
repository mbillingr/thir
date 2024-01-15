use crate::ambiguity::{default_subst, defaulted_preds};
use crate::assumptions::{find, Assump};
use crate::classes::ClassEnv;
use crate::kinds::Kind;
use crate::lists::{eq_diff, eq_intersect, eq_union, rfold1};
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::substitutions::{Subst, Types};
use crate::type_inference::TI;
use crate::types::{Type, Tyvar};
use crate::{Id, Int};
use std::iter::once;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Literal {
    Int(i64),
    Char(char),
    Rat(f64),
    Str(String),
}

fn ti_lit(ti: &mut TI, l: &Literal) -> crate::Result<(Vec<Pred>, Type)> {
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

#[derive(Clone, Debug)]
pub enum Pat {
    PVar(Id),
    PWildcard,
    PAs(Id, Rc<Pat>),
    PLit(Literal),
    PNpk(Id, Int),
    PCon(Assump, Vec<Pat>),
}

impl Types for Pat {
    fn apply_subst(&self, s: &Subst) -> Self {
        self.clone()
    }

    fn tv(&self) -> Vec<Tyvar> {
        vec![]
    }
}

fn ti_pat(ti: &mut TI, pat: &Pat) -> crate::Result<(Vec<Pred>, Vec<Assump>, Type)> {
    match pat {
        Pat::PVar(i) => {
            let v = ti.new_tvar(Kind::Star);
            Ok((
                vec![],
                vec![Assump {
                    i: i.clone(),
                    sc: v.clone().to_scheme(),
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
                sc: t.clone().to_scheme(),
            });

            Ok((ps, as_, t))
        }

        Pat::PLit(li) => {
            let (ps, t) = ti_lit(ti, li)?;
            Ok((ps, vec![], t))
        }

        Pat::PNpk(i, _) => {
            let t = ti.new_tvar(Kind::Star);
            Ok((
                vec![Pred::IsIn("Integral".into(), t.clone())],
                vec![Assump {
                    i: i.clone(),
                    sc: t.clone().to_scheme(),
                }],
                t,
            ))
        }

        Pat::PCon(Assump { sc, .. }, pats) => {
            let (mut ps, as_, ts) = ti_pats(ti, pats)?;
            let t_ = ti.new_tvar(Kind::Star);
            let Qual(qs, t) = ti.fresh_inst(sc);
            let f = ts
                .into_iter()
                .rfold(t_.clone(), |acc, t__| Type::func(t__, acc));
            ti.unify(&t, &f)?;
            ps.extend(qs);
            Ok((ps, as_, t_))
        }
    }
}

fn ti_pats(ti: &mut TI, pats: &[Pat]) -> crate::Result<(Vec<Pred>, Vec<Assump>, Vec<Type>)> {
    let psats = pats
        .iter()
        .map(|p| ti_pat(ti, p))
        .collect::<crate::Result<Vec<_>>>()?;

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

#[derive(Clone, Debug)]
pub enum Expr {
    Var(Id),
    Lit(Literal),
    Const(Assump),
    App(Rc<Expr>, Rc<Expr>),
    Let(BindGroup, Rc<Expr>),
    Annotate(Type, Rc<Expr>),
}

impl Types for Expr {
    fn apply_subst(&self, s: &Subst) -> Self {
        match self {
            Expr::Var(_) | Expr::Lit(_) | Expr::Const(_) => self.clone(),
            Expr::App(l, r) => Expr::App(l.apply_subst(s).into(), r.apply_subst(s).into()),
            Expr::Let(bg, e) => Expr::Let(bg.apply_subst(s), e.apply_subst(s).into()),
            Expr::Annotate(t, e) => Expr::Annotate(t.apply_subst(s), e.apply_subst(s).into()),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Expr::Var(_) | Expr::Lit(_) | Expr::Const(_) => vec![],
            Expr::App(l, r) => eq_union(l.tv(), r.tv()),
            Expr::Let(bg, e) => eq_union(bg.tv(), e.tv()),
            Expr::Annotate(t, e) => eq_union(t.tv(), e.tv()),
        }
    }
}

fn ti_expr(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    expr: &Expr,
) -> crate::Result<(Expr, Vec<Pred>, Type)> {
    match expr {
        Expr::Var(i) => {
            let sc = find(i, ass)?;
            let Qual(ps, t) = ti.fresh_inst(sc);
            Ok((
                Expr::Annotate(t.clone(), Expr::Var(i.clone()).into()),
                ps,
                t,
            ))
        }

        Expr::Const(Assump { sc, .. }) => {
            let Qual(ps, t) = ti.fresh_inst(sc);
            Ok((Expr::Annotate(t.clone(), expr.clone().into()), ps, t))
        }

        Expr::Lit(li) => {
            let (ps, t) = ti_lit(ti, li)?;
            Ok((Expr::Annotate(t.clone(), expr.clone().into()), ps, t))
        }

        Expr::App(e, f) => {
            let (f_, mut ps, te) = ti_expr(ti, ce, ass, e)?;
            let (a_, qs, tf) = ti_expr(ti, ce, ass, f)?;
            let t = ti.new_tvar(Kind::Star);
            ti.unify(&Type::func(tf, t.clone()), &te)?;
            ps.extend(qs);
            Ok((
                Expr::Annotate(t.clone(), Expr::App(f_.into(), a_.into()).into()),
                ps,
                t,
            ))
        }

        Expr::Let(bg, e) => {
            let (bg_, mut ps, mut ass_) = ti_bindgroup(ti, ce, ass, bg)?;
            ass_.extend(ass.iter().cloned());
            let (e_, qs, t) = ti_expr(ti, ce, &ass_, e)?;
            ps.extend(qs);
            Ok((Expr::Let(bg_, e_.into()), ps, t))
        }

        Expr::Annotate(t, ex) => {
            let (ex_, ps, t_) = ti_expr(ti, ce, ass, ex)?;
            ti.unify(t, &t_)?;
            Ok((ex_, ps, t_))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Alt(pub Vec<Pat>, pub Expr);

impl Types for Alt {
    fn apply_subst(&self, s: &Subst) -> Self {
        Alt(self.0.apply_subst(s), self.1.apply_subst(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        let tvs = self.0.tv();
        eq_union(tvs, self.1.tv())
    }
}

fn ti_alt(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    Alt(pats, e): &Alt,
) -> crate::Result<(Alt, Vec<Pred>, Type)> {
    let (mut ps, mut ass_, ts) = ti_pats(ti, pats)?;
    ass_.extend(ass.iter().cloned());
    let (e_, qs, t) = ti_expr(ti, ce, &ass_, e)?;
    ps.extend(qs);
    let f = ts.into_iter().rfold(t, |acc, t_| Type::func(t_, acc));
    Ok((Alt(pats.clone(), e_), ps, f))
}

fn ti_alts(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    alts: &[Alt],
    t: &Type,
) -> crate::Result<(Vec<Alt>, Vec<Pred>)> {
    let psts = alts
        .iter()
        .map(|a| ti_alt(ti, ce, ass, a))
        .collect::<crate::Result<Vec<_>>>()?;
    let mut alts_ = Vec::with_capacity(psts.len());
    let mut ps = vec![];
    for (a_, ps_, t_) in psts {
        alts_.push(a_);
        ti.unify(t, &t_)?;
        ps.extend(ps_);
    }
    Ok((alts_, ps))
}

fn split(
    ce: &ClassEnv,
    fs: &[Tyvar],
    gs: &[Tyvar],
    ps: &[Pred],
) -> crate::Result<(Vec<Pred>, Vec<Pred>)> {
    let ps_ = ce.reduce(ps)?;
    let (ds, rs): (Vec<_>, _) = ps_
        .into_iter()
        .partition(|p| p.tv().iter().all(|tv| fs.contains(tv)));
    let mut fsgs = vec![];
    fsgs.extend(fs.iter().chain(gs.iter()).cloned());
    let rs_ = defaulted_preds(ce, fsgs, &rs)?;
    Ok((ds, eq_diff(rs, rs_)))
}

/// Explicitly typed binding
#[derive(Clone, Debug)]
pub struct Expl(pub Id, pub Scheme, pub Vec<Alt>);

impl Types for Expl {
    fn apply_subst(&self, s: &Subst) -> Self {
        // todo: not sure if we should substitute the scheme here, since the purpose of this substitution is to
        //       resolve type variables in the bodies of the declarations.
        Expl(self.0.clone(), self.1.apply_subst(s), self.2.apply_subst(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        // todo: not sure if we should include the scheme's type vars...
        eq_union(self.1.tv(), self.2.tv())
    }
}

pub fn ti_expl(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    Expl(id, sc, alts): &Expl,
) -> crate::Result<(Expl, Vec<Pred>)> {
    let Qual(qs, t) = ti.fresh_inst(sc);
    let (alts_, ps) = ti_alts(ti, ce, ass, alts, &t)?;
    let s = &ti.get_subst();
    let qs_ = s.apply(&qs);
    let t_ = s.apply(&t);
    let fs = s.apply(ass).tv();
    let gs = eq_diff(t_.tv(), fs.clone());
    let ps_: Vec<_> = s
        .apply(&ps)
        .into_iter()
        .filter(|p| !ce.entail(&qs_, p))
        .collect();
    let sc_ = Scheme::quantify(&gs, &Qual(qs_.clone(), t_.clone()));
    let (ds, rs) = split(ce, &fs, &gs, &ps_)?;

    if sc != &sc_ {
        Err(format!("signature too general {sc:?} != {sc_:?}"))?;
    }

    if !rs.is_empty() {
        Err("context too weak")?;
    }

    // this part was added to substitute type vars in annotations with actual types
    let rs__ = ce.reduce(&s.apply(&ps_))?;
    let s_ = default_subst(ce, vec![], &rs__)?;
    let s__ = Scheme::quantifying_substitution(&gs, &Qual(qs_, t_));
    let alts_ = alts_.apply_subst(&s__.compose(&s_.compose(&s)));

    let expl_ = Expl(id.clone(), sc.clone(), alts_);

    Ok((expl_, ds))
}

/// Implicitly typed binding
#[derive(Clone, Debug)]
pub struct Impl(pub Id, pub Vec<Alt>);

impl Types for Impl {
    fn apply_subst(&self, s: &Subst) -> Self {
        Impl(self.0.clone(), self.1.apply_subst(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        self.1.tv()
    }
}

fn restricted(bs: &[Impl]) -> bool {
    fn simple(Impl(_, alts): &Impl) -> bool {
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
) -> crate::Result<(Vec<Impl>, Vec<Pred>, Vec<Assump>)> {
    let ts: Vec<_> = bs.iter().map(|_| ti.new_tvar(Kind::Star)).collect();
    let is = || bs.iter().map(|Impl(i, _)| i.clone());
    let scs = ts.iter().cloned().map(Type::to_scheme);
    let as_: Vec<_> = is()
        .zip(scs)
        .map(|(i, sc)| Assump { i, sc })
        .chain(ass.iter().cloned())
        .collect();
    let altss = bs.iter().map(|Impl(_, alts)| alts);

    let a_pss = altss
        .zip(&ts)
        .map(|(a, t)| ti_alts(ti, ce, &as_, a, t))
        .collect::<crate::Result<Vec<_>>>()?;
    let mut altss_ = Vec::with_capacity(a_pss.len());
    let mut pss = Vec::with_capacity(a_pss.len());
    for (a_, p_) in a_pss {
        altss_.push(a_);
        pss.push(p_);
    }

    let s = &ti.get_subst();
    let ps_ = s.apply(&pss.into_iter().flatten().collect::<Vec<_>>());
    let ts_ = s.apply(&ts);
    let fs = s.apply(ass).tv();
    let vss = || ts_.iter().map(Types::tv);
    let (mut ds, rs) = split(ce, &fs, &rfold1(vss(), eq_intersect), &ps_)?;
    let gs = eq_diff(rfold1(vss(), eq_union), fs);

    let bs_ = bs
        .iter()
        .zip(altss_)
        .map(|(Impl(id, _), alts_)| Impl(id.clone(), alts_))
        .collect();

    // todo: substitute annotations (see ti_expl)

    if restricted(bs) {
        let gs_ = eq_diff(gs, rs.tv());
        let scs_ = ts_
            .into_iter()
            .map(|t| Scheme::quantify(&gs_, &Qual(vec![], t)));
        ds.extend(rs);
        Ok((
            bs_,
            ds,
            is().zip(scs_).map(|(i, sc)| Assump { i, sc }).collect(),
        ))
    } else {
        let scs_ = ts_
            .into_iter()
            .map(|t| Scheme::quantify(&gs, &Qual(rs.clone(), t)));
        Ok((
            bs_,
            ds,
            is().zip(scs_).map(|(i, sc)| Assump { i, sc }).collect(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct BindGroup(pub Vec<Expl>, pub Vec<Vec<Impl>>);

impl Types for BindGroup {
    fn apply_subst(&self, s: &Subst) -> Self {
        BindGroup(self.0.apply_subst(s), self.1.apply_subst(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        let tvs = self.0.tv();
        self.1.iter().map(|imps| imps.tv()).fold(tvs, eq_union)
    }
}

fn ti_bindgroup(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    BindGroup(es, iss): &BindGroup,
) -> crate::Result<(BindGroup, Vec<Pred>, Vec<Assump>)> {
    let as1: Vec<_> = es
        .iter()
        .map(|Expl(v, sc, _)| Assump {
            i: v.clone(),
            sc: sc.clone(),
        })
        .collect();

    let mut as1_as = as1.clone();
    as1_as.extend(ass.to_vec());

    let (iss_, ps, as2) = ti_seq(ti_impls, ti, ce, as1_as, iss)?;

    let mut as2_as1 = as2.clone();
    as2_as1.extend(as1);

    let mut as2_as1_as = as2_as1.clone();
    as2_as1_as.extend(ass.to_vec());

    let es_qss = es
        .iter()
        .map(|e| ti_expl(ti, ce, &as2_as1_as, e))
        .collect::<crate::Result<Vec<_>>>()?;

    let mut es_ = Vec::with_capacity(es_qss.len());
    let mut qss = Vec::with_capacity(es_qss.len());
    for (e, q) in es_qss {
        es_.push(e);
        qss.push(q);
    }

    let bg_ = BindGroup(es_, iss_);

    Ok((bg_, once(ps).chain(qss).flatten().collect(), as2_as1))
}

fn ti_seq<T>(
    inf: impl Fn(&mut TI, &ClassEnv, &[Assump], &T) -> crate::Result<(T, Vec<Pred>, Vec<Assump>)>,
    ti: &mut TI,
    ce: &ClassEnv,
    ass: Vec<Assump>,
    bss: &[T],
) -> crate::Result<(Vec<T>, Vec<Pred>, Vec<Assump>)> {
    if bss.is_empty() {
        return Ok((vec![], vec![], vec![]));
    }

    let bs = &bss[0];
    let bss = &bss[1..];

    let (bs_, mut ps, as_) = inf(ti, ce, &ass, bs)?;

    let mut as_as = as_.clone();
    as_as.extend(ass);

    let (mut bss_, qs, mut as__) = ti_seq(inf, ti, ce, as_as, bss)?;

    bss_.insert(0, bs_);
    ps.extend(qs);
    as__.extend(as_);
    Ok((bss_, ps, as__))
}

#[derive(Debug)]
pub struct Program(pub Vec<BindGroup>);

pub fn ti_program(
    ce: &ClassEnv,
    ass: Vec<Assump>,
    Program(bgs): &Program,
) -> crate::Result<(Program, Vec<Assump>)> {
    let mut ti = TI::new();
    let (bgs_, ps, as_) = ti_seq(ti_bindgroup, &mut ti, ce, ass, bgs)?;
    //println!("{:#?}", ps);
    //println!("{:#?}", as_);
    let s = &ti.get_subst();
    let rs = ce.reduce(&s.apply(&ps))?;
    let s_ = default_subst(ce, vec![], &rs)?;
    println!("{:?}", ti);
    Ok((Program(bgs_), s_.compose(s).apply(&as_)))
}
