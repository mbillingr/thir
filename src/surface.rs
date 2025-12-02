//! the surface language

use crate::assumptions::Assump;
pub use crate::kinds::Kind;
pub use crate::specific_inference::Literal;
use crate::{predicates, qualified, scheme, specific_inference, types, Id};
use im_rc::HashMap;
use std::rc::Rc;

pub struct Program {
    cls_defs: Vec<ClassDef>,
    ty_defs: Vec<TypeDef>,
    cls_impls: Vec<ClassImpl>,
    code: Vec<BindGroup>
}

struct ClassDef {
    // todo
}

struct TypeDef {
    // todo
}

struct ClassImpl {
    // todo
}

pub struct BindGroup {
    expls: Vec<Expl>,
    impls: Vec<Impl>,
}

struct Impl {
    name: Id,
    alts: Vec<Alt>,
}

struct Expl {
    name: Id,
    scm: Scheme,
    alts: Vec<Alt>,
}

struct Alt {
    pats: Vec<Pat>,
    body: Expr,
}

pub enum Type {
    TRef(Id),
    TApp(Box<(Type, Type)>),
}

pub struct Scheme {
    args: Vec<TyArg>,
    body: Type,
}

pub struct TyArg {
    name: Id,
    kind: Kind,
    type_classes: Vec<Id>,
}

pub enum Pat {
    Wildcard,
    Var(Id),
    Alias(Id, Box<Pat>),
    Literal(Literal),
    Constructor(Id, Vec<Pat>),
}

pub enum Expr {
    Var(Id),
    Lit(Literal),
    App(Box<(Expr, Expr)>),
    Let(BindGroup, Box<Expr>),
}

#[derive(Clone)]
struct Context {
    types: HashMap<Id, types::Type>,
    constructors: HashMap<Id, Assump>,
}

impl Context {
    fn with_type_params(&self, params: &[TyArg]) -> Context {
        let mut ctx = self.clone();

        for (i, arg) in params.iter().enumerate() {
            ctx.types.insert(arg.name.clone(), types::Type::TGen(i));
        }

        ctx
    }
}

trait IntoTck<T> {
    fn into_tck(self, ctx: &Context) -> T;
}

impl IntoTck<specific_inference::BindGroup> for BindGroup {
    fn into_tck(self, ctx: &Context) -> specific_inference::BindGroup {
        specific_inference::BindGroup(self.expls.into_tck(ctx), vec![self.impls.into_tck(ctx)])
    }
}

impl IntoTck<specific_inference::Impl> for Impl {
    fn into_tck(self, ctx: &Context) -> specific_inference::Impl {
        specific_inference::Impl(self.name, self.alts.into_tck(ctx))
    }
}

impl IntoTck<specific_inference::Expl> for Expl {
    fn into_tck(self, ctx: &Context) -> specific_inference::Expl {
        specific_inference::Expl(self.name, self.scm.into_tck(ctx), self.alts.into_tck(ctx))
    }
}

impl IntoTck<specific_inference::Alt> for Alt {
    fn into_tck(self, ctx: &Context) -> specific_inference::Alt {
        specific_inference::Alt(self.pats.into_tck(ctx), self.body.into_tck(ctx))
    }
}

impl IntoTck<scheme::Scheme> for Scheme {
    fn into_tck(self, ctx: &Context) -> scheme::Scheme {
        let local_ctx = ctx.with_type_params(&self.args);
        let ty = self.body.into_tck(&local_ctx);

        let kinds = self.args.iter().map(|arg| arg.kind.clone()).collect();
        let preds = self
            .args
            .into_iter()
            .enumerate()
            .flat_map(|(i, arg)| {
                arg.type_classes
                    .into_iter()
                    .map(move |tc| predicates::Pred::IsIn(tc, types::Type::TGen(i)))
            })
            .collect();

        scheme::Scheme::Forall(kinds, qualified::Qual(preds, ty))
    }
}

impl IntoTck<types::Type> for Type {
    fn into_tck(self, ctx: &Context) -> types::Type {
        match self {
            Type::TRef(id) => ctx.types.get(&id).expect("unbound type").clone(),
            Type::TApp(app) => {
                types::Type::TApp(Rc::new((app.0.into_tck(ctx), app.1.into_tck(ctx))))
            }
        }
    }
}

impl IntoTck<specific_inference::Pat> for Pat {
    fn into_tck(self, ctx: &Context) -> specific_inference::Pat {
        match self {
            Pat::Wildcard => specific_inference::Pat::PWildcard,
            Pat::Var(name) => specific_inference::Pat::PVar(name),
            Pat::Alias(name, pat) => specific_inference::Pat::PAs(name, Rc::new(pat.into_tck(ctx))),
            Pat::Literal(literal) => specific_inference::Pat::PLit(literal),
            Pat::Constructor(name, sub_pats) => specific_inference::Pat::PCon(
                ctx.constructors
                    .get(&name)
                    .expect("unbound constructor")
                    .clone(),
                sub_pats.into_tck(ctx),
            ),
        }
    }
}

impl IntoTck<specific_inference::Expr> for Expr {
    fn into_tck(self, ctx: &Context) -> specific_inference::Expr {
        match self {
            Expr::Var(name) => specific_inference::Expr::Var(name),
            Expr::Lit(lit) => specific_inference::Expr::Lit(lit),
            Expr::App(app) => specific_inference::Expr::App(Rc::new(app.0.into_tck(ctx)), Rc::new(app.1.into_tck(ctx))),
            Expr::Let(bg, body) => specific_inference::Expr::Let(bg.into_tck(ctx), Rc::new(body.into_tck(ctx))),
        }
    }
}

impl<U, T: IntoTck<U>> IntoTck<Vec<U>> for Vec<T> {
    fn into_tck(self, ctx: &Context) -> Vec<U> {
        self.into_iter().map(|t| t.into_tck(ctx)).collect()
    }
}
