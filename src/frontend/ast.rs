//! The AST emitted by the parser.

pub use crate::frontend::type_inference::Literal;
use crate::type_checker::kinds::Kind;
use crate::type_checker::Id;
use std::rc::Rc;

#[derive(Debug)]
pub enum TopLevel {
    Include(Rc<str>),
    DefClass(DefClass),
    ImplClass(ImplClass),
    DataType(DataType),
    BindGroup(BindGroup),
}

#[derive(Debug)]
pub struct DefClass {
    pub name: Id,
    pub varname: Id,
    pub super_classes: Vec<Id>,
    pub methods: Vec<(Id, Scheme)>,
}

impl DefClass {
    pub fn new(name: Id, varname: Id, super_classes: Vec<Id>, methods: Vec<(Id, Scheme)>) -> Self {
        DefClass {
            name,
            varname,
            super_classes,
            methods,
        }
    }
}

#[derive(Debug)]
pub struct ImplClass {
    pub cls: Id,
    pub genvars: Vec<(Id, Kind, Vec<Id>)>,
    pub ty: Type,
    pub methods: Vec<Impl>,
}

impl ImplClass {
    pub fn new(cls: Id, genvars: Vec<(Id, Kind, Vec<Id>)>, ty: Type, methods: Vec<Impl>) -> Self {
        ImplClass {
            cls,
            genvars,
            ty,
            methods,
        }
    }

    pub fn new_specific(cls: Id, ty: Type, methods: Vec<Impl>) -> Self {
        Self::new(cls, vec![], ty, methods)
    }
}

#[derive(Debug)]
pub struct DataType {
    pub typename: Id,
    pub genvars: Vec<(Id, Kind, Vec<Id>)>,
    pub constructors: Vec<(Id, Vec<Type>)>,
}

impl DataType {
    pub fn new(
        typename: Id,
        genvars: Vec<(Id, Kind, Vec<Id>)>,
        constructors: Vec<(Id, Vec<Type>)>,
    ) -> Self {
        DataType {
            genvars,
            typename,
            constructors,
        }
    }
}

#[derive(Debug)]
pub enum Pat {
    PVar(Id),
    PWildcard,
    PAs(Id, Box<Pat>),
    PLit(Literal),
    PNpk(Id, i64),
    PCon(Id, Vec<Pat>),
}

impl Pat {
    pub fn pas(p: Pat, i: Id) -> Pat {
        Pat::PAs(i, Box::new(p))
    }
}

#[derive(Debug)]
pub enum Expr {
    Var(Id),
    Lit(Literal),
    App(Box<Expr>, Box<Expr>),
    Let(BindGroup, Box<Expr>),
    Seq(Vec<Expr>, Box<Expr>),

    Infix(Vec<InfixToken>),

    Lambda(Box<Alt>),
}

#[derive(Debug)]
pub enum InfixToken {
    Expr(Expr),
    Op(Id),
}

impl Expr {
    pub fn app(a: Expr, b: Expr) -> Expr {
        Expr::App(Box::new(a), Box::new(b))
    }

    pub fn let_(bg: BindGroup, e: Expr) -> Expr {
        Expr::Let(bg, Box::new(e))
    }

    pub fn sequence(stmts: Vec<Expr>, last: Expr) -> Expr {
        Expr::Seq(stmts, Box::new(last))
    }
}

#[derive(Debug)]
pub struct Alt(pub Vec<Pat>, pub Expr);

#[derive(Debug)]
pub struct Decl(pub Id, pub Scheme);

#[derive(Debug)]
pub struct Impl(pub Id, pub Vec<Alt>);

#[derive(Debug)]
pub enum Bind {
    Declaration(Decl),
    Implicit(Impl),
    Mutual(Vec<Impl>),
}

#[derive(Debug)]
pub struct BindGroup(pub Vec<Bind>);

#[derive(Clone, Debug)]
pub enum Type {
    Named(Id),
    Apply(Box<Type>, Box<Type>),
}

impl Type {
    pub fn apply(a: Type, b: Type) -> Type {
        Type::Apply(Box::new(a), Box::new(b))
    }

    pub fn func(a: Type, b: Type) -> Type {
        Type::Apply(
            Box::new(Type::Apply(Box::new(Type::Named("->".into())), Box::new(a))),
            Box::new(b),
        )
    }

    pub fn list(a: Type) -> Type {
        Type::Apply(Box::new(Type::Named("[]".into())), Box::new(a))
    }
}

#[derive(Clone, Debug)]
pub struct Scheme {
    pub genvars: Vec<(Id, Kind, Vec<Id>)>,
    pub ty: Type,
}

impl Scheme {
    pub fn new(genvars: Vec<(Id, Kind, Vec<Id>)>, ty: Type) -> Self {
        Scheme { genvars, ty }
    }
}

pub fn unescape(s: &str) -> Rc<str> {
    let mut out = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('0') => out.push('\0'),
                Some(c) => out.push(c),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }

    out.into()
}
