use crate::specific_inference;
use crate::specific_inference::Literal;
use chumsky::prelude::{todo, Spanned};
use ustr::Ustr;

#[derive(Debug)]
pub enum TopLevel {
    TypeDef(Spanned<TypeDef>),
    Expr(Spanned<Expr>),
}

#[derive(Debug)]
pub struct TypeDef {
    pub tname: Spanned<TypeName>,
    pub params: Vec<Spanned<TypeVar>>,
    pub constraints: Vec<Spanned<Constraint>>,
    pub variants: Vec<Spanned<VariantDef>>,
}

#[derive(Debug)]
pub struct VariantDef {
    pub name: Spanned<ConstructorName>,
    pub fields: Vec<Spanned<TExpr>>,
}

#[derive(Debug)]
pub struct Constraint {
    pub cls: Spanned<ClassName>,
    pub tys: Vec<Spanned<TExpr>>,
}

#[derive(Debug)]
pub struct ClassName(pub Ustr);

#[derive(Debug)]
pub struct TypeName(pub Ustr);

#[derive(Debug)]
pub struct ConstructorName(pub Ustr);

#[derive(Debug)]
pub struct TypeVar(pub Ustr);

#[derive(Debug)]
pub enum TExpr {
    Sym(Ustr),
    App(Box<Spanned<TExpr>>, Box<Spanned<TExpr>>),
}

#[derive(Debug)]
pub enum Expr {
    Literal(Literal),
    Var(Ustr),
    App(Vec<Spanned<Expr>>),
}

pub fn convert_expression(expr: &Spanned<Expr>) -> specific_inference::Expr {
    match &expr.inner {
        Expr::Literal(lit) => specific_inference::Expr::Lit(lit.clone()),
        Expr::Var(name) => specific_inference::Expr::Var(name.clone()),
        Expr::App(xs) => match xs.as_slice() {
            [] => unimplemented!(),
            [f] => todo!(),
            [f, a] => specific_inference::Expr::App(
                convert_expression(f).into(),
                convert_expression(a).into(),
            ),
            [f, args @ ..] => todo!(),
        },
        _ => todo!(),
    }
}
