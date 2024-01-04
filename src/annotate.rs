use crate::assumptions::Assump;
use crate::instantiate::Instantiate;
use crate::kinds::Kind;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{Alt, Expr, Pat};
use crate::type_inference::TI;
use crate::types::{Tycon, Type};
use crate::Id;

pub fn annotate_function(id: &Id, alts: &[Alt], targs: &[Type], ass: &Assump) -> Vec<Alt> {
    assert_eq!(id, &ass.i);

    let mut ti = TI::new();
    let Qual(ps, t) = ti.fresh_inst(&ass.sc);

    let signature = match &ass.sc {
        Scheme::Forall(ks, qt) => {
            let Qual(ps, t) = qt.inst(targs);
            //todo: check predicates
            t
        }
    };

    alts.iter().map(|a| annotate_alt(a, &signature)).collect()
}

pub fn annotate_alt(Alt(pats, expr): &Alt, signature: &Type) -> Alt {
    println!("{:?}", signature);
    let (env, tres) = match_pats(pats, signature, Env::empty());
    println!("{:?}", env);
    todo!()
}

fn match_pats<'a>(pats: &[Pat], sig: &'a Type, env: Env) -> (Env, &'a Type) {
    match (pats, sig) {
        ([], t) => (env, t),
        ([Pat::PWildcard, ps @ ..], Type::TApp(_)) => {
            let (_, res) = dbg!(expect_func_type(sig));
            match_pats(ps, res, env)
        }
        ([Pat::PVar(pv), ps @ ..], Type::TApp(_)) => {
            let (arg, res) = dbg!(expect_func_type(sig));
            match_pats(ps, res, env.extend(pv.clone(), arg.clone()))
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

#[derive(Debug)]
struct Env(Vec<(Id, Type)>);

impl Env {
    fn empty() -> Self {
        Env(vec![])
    }

    fn extend(mut self, name: Id, ty: Type) -> Self {
        self.0.push((name, ty));
        self
    }
}
