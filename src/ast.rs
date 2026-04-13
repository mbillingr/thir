use crate::specific_inference;
use crate::specific_inference::Literal;
use chumsky::prelude::{todo, Spanned};
use ustr::Ustr;

#[derive(Debug)]
pub enum TopLevel {
    TypeDef(Spanned<TypeDef>),
    ClassDef(Spanned<ClassDef>),
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
pub struct ClassDef {
    pub cname: Spanned<ClassName>,
    pub params: Vec<Spanned<TypeVar>>,
    pub supers: Vec<Spanned<Constraint>>,
    pub methods: Vec<Spanned<Declaration>>,
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
pub struct Declaration {
    pub name: Spanned<Variable>,
    pub ty: Spanned<TExpr>,
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
pub struct Variable(pub Ustr);

#[derive(Debug)]
pub enum TExpr {
    Sym(Ustr),
    App(Box<Spanned<TExpr>>, Box<Spanned<TExpr>>),
}

#[derive(Debug)]
pub enum Expr {
    Literal(Literal),
    Var(Variable),
    App(Vec<Spanned<Expr>>),
}

pub fn convert_expression(expr: &Spanned<Expr>) -> specific_inference::Expr {
    match &expr.inner {
        Expr::Literal(lit) => specific_inference::Expr::Lit(lit.clone()),
        Expr::Var(Variable(name)) => specific_inference::Expr::Var(name.clone()),
        Expr::App(xs) => match xs.as_slice() {
            [] => unimplemented!(),
            [f] => todo!(),
            [f, a] => specific_inference::Expr::App(
                convert_expression(f).into(),
                convert_expression(a).into(),
            ),
            [f, args @ ..] => {
                let mut expr = convert_expression(f);
                for arg in args {
                    expr =
                        specific_inference::Expr::App(expr.into(), convert_expression(arg).into());
                }
                expr
            }
        },
        _ => todo!(),
    }
}
