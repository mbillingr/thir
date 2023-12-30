use crate::assumptions::Assump;
use crate::classes::ClassEnv;
use crate::kinds::Kind;
use crate::lists::{list_diff, list_intersect, list_union, rfold1};
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::substitutions::Types;
use crate::type_inference::TI;
use crate::types::{Type, Tyvar};
use crate::{default_subst, defaulted_preds, find, quantify, to_scheme, Id, Int};
use std::iter::once;
use std::rc::Rc;

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

pub enum Pat {
    PVar(Id),
    PWildcard,
    PAs(Id, Rc<Pat>),
    PLit(Literal),
    PNpk(Id, Int),
    PCon(Assump, Vec<Pat>),
}

fn ti_pat(ti: &mut TI, pat: &Pat) -> crate::Result<(Vec<Pred>, Vec<Assump>, Type)> {
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

pub enum Expr {
    Var(Id),
    Lit(Literal),
    Const(Assump),
    App(Rc<Expr>, Rc<Expr>),
    Let(BindGroup, Rc<Expr>),
}

fn ti_expr(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    expr: &Expr,
) -> crate::Result<(Vec<Pred>, Type)> {
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

pub struct Alt(pub Vec<Pat>, pub Expr);

fn ti_alt(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    Alt(pats, e): &Alt,
) -> crate::Result<(Vec<Pred>, Type)> {
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
) -> crate::Result<Vec<Pred>> {
    let psts = alts
        .iter()
        .map(|a| ti_alt(ti, ce, ass, a))
        .collect::<crate::Result<Vec<_>>>()?;
    let mut ps = vec![];
    for (ps_, t_) in psts {
        ti.unify(t, &t_)?;
        ps.extend(ps_);
    }
    Ok(ps)
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
    Ok((ds, list_diff(rs, rs_)))
}

/// Explicitly typed binding
pub struct Expl(pub Id, pub Scheme, pub Vec<Alt>);

fn ti_expl(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    Expl(i, sc, alts): &Expl,
) -> crate::Result<Vec<Pred>> {
    let Qual(qs, t) = ti.fresh_inst(sc);
    let ps = ti_alts(ti, ce, ass, alts, &t)?;
    let s = &ti.get_subst();
    let qs_ = s.apply(&qs);
    let t_ = s.apply(&t);
    let fs = s.apply(ass).tv();
    let gs = list_diff(t_.tv(), fs.clone());
    let ps_: Vec<_> = s
        .apply(&ps)
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
pub struct Impl(pub Id, pub Vec<Alt>);

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
) -> crate::Result<(Vec<Pred>, Vec<Assump>)> {
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
        .collect::<crate::Result<Vec<_>>>()?;
    let s = &ti.get_subst();
    let ps_ = s.apply(&pss.into_iter().flatten().collect::<Vec<_>>());
    let ts_ = s.apply(&ts);
    let fs = s.apply(ass).tv();
    let vss = || ts_.iter().map(Types::tv);
    let (mut ds, rs) = split(ce, &fs, &rfold1(vss(), list_intersect), &ps_)?;
    let gs = list_diff(rfold1(vss(), list_union), fs);
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

pub struct BindGroup(pub Vec<Expl>, pub Vec<Vec<Impl>>);

fn ti_bindgroup(
    ti: &mut TI,
    ce: &ClassEnv,
    ass: &[Assump],
    BindGroup(es, iss): &BindGroup,
) -> crate::Result<(Vec<Pred>, Vec<Assump>)> {
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
        .collect::<crate::Result<Vec<_>>>()?;

    Ok((once(ps).chain(qss).flatten().collect(), as__as_))
}

fn ti_seq<T>(
    inf: impl Fn(&mut TI, &ClassEnv, &[Assump], &T) -> crate::Result<(Vec<Pred>, Vec<Assump>)>,
    ti: &mut TI,
    ce: &ClassEnv,
    ass: Vec<Assump>,
    bss: &[T],
) -> crate::Result<(Vec<Pred>, Vec<Assump>)> {
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

pub struct Program(pub Vec<BindGroup>);

pub fn ti_program(
    ce: &ClassEnv,
    ass: Vec<Assump>,
    Program(bgs): &Program,
) -> crate::Result<Vec<Assump>> {
    let mut ti = TI::new();
    let (ps, as_) = ti_seq(ti_bindgroup, &mut ti, ce, ass, bgs)?;
    //println!("{:#?}", ps);
    //println!("{:#?}", as_);
    let s = &ti.get_subst();
    let rs = ce.reduce(&s.apply(&ps))?;
    let s_ = default_subst(ce, vec![], &rs)?;
    Ok(s_.compose(s).apply(&as_))
}
