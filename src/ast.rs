use crate::kinds::Kind;
pub use crate::specific_inference::Literal;
use crate::Id;

#[derive(Debug)]
pub enum TopLevel {
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
    pub ty: Id,
    pub methods: Vec<Impl>,
}

impl ImplClass {
    pub fn new(cls: Id, ty: Id, methods: Vec<Impl>) -> Self {
        ImplClass { cls, ty, methods }
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
    //PCon(Assump, Vec<Pat>),  //todo: figure this one out
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
}

impl Expr {
    pub fn app(a: Expr, b: Expr) -> Expr {
        Expr::App(Box::new(a), Box::new(b))
    }

    pub fn let_(bg: BindGroup, e: Expr) -> Expr {
        Expr::Let(bg, Box::new(e))
    }
}

#[derive(Debug)]
pub struct Alt(pub Vec<Pat>, pub Expr);

#[derive(Debug)]
pub struct Expl(pub Id, pub Scheme, pub Vec<Alt>);

#[derive(Debug)]
pub struct Impl(pub Id, pub Vec<Alt>);

#[derive(Debug)]
pub enum Bind {
    Explicit(Expl),
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
