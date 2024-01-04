use crate::assumptions::Assump;
use crate::instantiate::Instantiate;
use crate::kinds::Kind;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{Alt, Expr, Pat};
use crate::type_inference::TI;
use crate::types::{Tycon, Type};
use crate::Id;
use std::borrow::Cow;

pub fn annotate_function(id: &Id, alts: &[Alt], targs: &[Type], ass: &[Assump]) -> Vec<Alt> {
    let env = Env(Cow::Borrowed(ass));
    let sc = env.lookup(id);

    let mut ti = TI::new();
    let Qual(ps, t) = ti.fresh_inst(sc);

    let signature = match sc {
        Scheme::Forall(ks, qt) => {
            let Qual(ps, t) = qt.inst(targs);
            //todo: check predicates
            t
        }
    };

    alts.iter()
        .map(|a| annotate_alt(a, &signature, env.borrowed()))
        .collect()
}

pub fn annotate_alt<'e>(Alt(pats, expr): &Alt, signature: &Type, env: Env<'e>) -> Alt {
    println!("{:?}", signature);
    let (env, tres) = match_pats(pats, signature, env);
    println!("{:?}", env);
    let expr_ = annotate_expr(expr, env);
    Alt(pats.clone(), expr_)
}

fn match_pats<'a, 'e>(pats: &[Pat], sig: &'a Type, env: Env<'e>) -> (Env<'e>, &'a Type) {
    match (pats, sig) {
        ([], t) => (env, t),

        ([Pat::PWildcard, ps @ ..], Type::TApp(_)) => {
            let (_, res) = dbg!(expect_func_type(sig));
            match_pats(ps, res, env)
        }

        ([Pat::PVar(pv), ps @ ..], Type::TApp(_)) => {
            let (arg, res) = dbg!(expect_func_type(sig));
            match_pats(ps, res, env.extend(pv.clone(), todo!()))
        }

        _ => todo!("{pats:?}, {sig:?}"),
    }
}

fn expect_func_type(sig: &Type) -> (&Type, &Type) {
    match sig {
        Type::TApp(ts) => match &ts.0 {
            Type::TApp(tt) => match &tt.0 {
                Type::TCon(Tycon(tc, _)) if tc == "->" => (&tt.1, &ts.1),
                _ => panic!("invalid signature"),
            },
            _ => panic!("invalid signature"),
        },
        _ => panic!("invalid signature"),
    }
}

fn annotate_expr(expr: &Expr, env: Env) -> Expr {
    match expr {
        Expr::Var(x) => {
            let sc = env.lookup(x);
            Expr::Annotation(todo!(), Expr::Var(x.clone()).into())
        }
        Expr::Lit(_) => todo!(),
        Expr::Const(_) => todo!(),
        Expr::App(f, a) => {
            let f_ = annotate_expr(f, env.borrowed());
            let a_ = annotate_expr(a, env);

            let tf = type_of(&f_);
            let ta = type_of(&a_);

            todo!("{tf:?}, {ta:?}")
        }
        Expr::Let(_, _) => todo!(),
        Expr::Annotation(_, _) => todo!(),
    }
}

fn type_of(expr: &Expr) -> &Type {
    match expr {
        Expr::Annotation(t, _) => t,
        _ => panic!("unannotated expression"),
    }
}

fn lookup<'a>(id: &Id, env: &'a [Assump]) -> &'a Scheme {
    env.iter()
        .rfind(|Assump { i, .. }| i == id)
        .map(|Assump { sc, .. }| sc)
        .unwrap_or_else(|| panic!("unbound {id}"))
}

#[derive(Debug)]
struct Env<'a>(Cow<'a, [Assump]>);

impl<'e> Env<'e> {
    fn borrowed(&self) -> Env {
        Env(match &self.0 {
            Cow::Borrowed(env) => Cow::Borrowed(*env),
            Cow::Owned(env) => Cow::Borrowed(env),
        })
    }

    fn empty() -> Self {
        Env(Cow::Owned(vec![]))
    }

    fn extend(self, i: Id, sc: Scheme) -> Self {
        let mut env = self.0.into_owned();
        env.push(Assump { i, sc });
        Env(Cow::Owned(env))
    }

    fn lookup(&self, name: &Id) -> &Scheme {
        self.0
            .iter()
            .rfind(|Assump { i, .. }| i == name)
            .map(|Assump { sc, .. }| sc)
            .unwrap_or_else(|| panic!("unbound {name}"))
    }
}
