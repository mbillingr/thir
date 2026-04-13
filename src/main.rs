// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

use crate::instantiate::Instantiate;
use crate::substitutions::Types;
use chumsky::input::Input;
use chumsky::Parser;
mod ambiguity;
mod assumptions;
mod ast;
mod classes;
mod instantiate;
mod kinds;
mod lists;
mod parsing_ast;
mod parsing_tokenize;
mod predicates;
mod qualified;
mod scheme;
mod specific_inference;
mod specifics;
mod substitutions;
mod type_inference;
mod types;
mod unification;

use crate::assumptions::Assump;
use crate::ast::convert_expression;
use crate::classes::{ClassEnv, EnvTransformer};
use crate::kinds::Kind;
use crate::parsing_ast::{toplevel, type_def, type_expr};
use crate::parsing_tokenize::lexer;
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{finalize, ti_expl, ti_expr, Alt, Expl, Expr, Literal, Pat};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::substitutions::Subst;
use crate::type_inference::TI;
use crate::types::{Tycon, Type, Tyvar};
use ariadne::{sources, Color, Label, Report, ReportKind};
use chumsky::error::Rich;
use chumsky::prelude::Spanned;
use sexpr_parser::{Parser as OtherParser, SexprFactory};
use std::collections::HashSet;
use std::fmt::Display;
use std::io::{self, BufRead, Write};
use std::rc::Rc;
use ustr::{ustr, Ustr};

type Result<T> = std::result::Result<T, String>;

fn main() {
    let mut ce = add_core_classes()
        .compose(add_num_classes())
        .apply(&ClassEnv::default())
        .unwrap();
    let mut cls_methods: im_rc::HashMap<Ustr, Vec<Ustr>> = Default::default();
    let mut ass: Vec<Assump> = vec![];
    let mut tenv: im_rc::HashMap<Ustr, Type> = im_rc::HashMap::new();

    tenv.insert(ustr("Int"), Type::t_int());
    tenv.insert(ustr("Float"), Type::t_float());
    tenv.insert(ustr("Double"), Type::t_double());
    tenv.insert(ustr("Symbol"), Type::t_symbol());
    tenv.insert(ustr("String"), Type::t_string());
    tenv.insert(ustr("->"), Type::t_arrow());

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut handle = stdin.lock();

    let mut buf = String::new();
    loop {
        println!("TYS: {:?}", tenv);
        println!("ASS: {:?}", ass);

        print!("> ");
        let _ = stdout.flush();

        buf.clear();
        loop {
            match handle.read_line(&mut buf) {
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    return;
                }
                Ok(0) => return, // EOF
                Ok(_) => {
                    if buf
                        .chars()
                        .filter_map(|ch| match ch {
                            '(' => Some(1),
                            ')' => Some(-1),
                            _ => None,
                        })
                        .sum::<isize>()
                        <= 0
                    {
                        break;
                    }
                }
            }
        }

        let tokens = match lexer().parse(&buf).into_result() {
            Ok(tokens) => tokens,
            Err(errs) => {
                report_errors(&buf, errs);
                continue;
            }
        };

        println!("{:?}", tokens);

        let top = match toplevel()
            .parse(tokens.split_spanned((0..buf.len()).into()))
            .into_result()
        {
            Ok(top) => top,
            Err(errs) => {
                report_errors(&buf, errs);
                continue;
            }
        };

        println!("{:?}", top);

        let res = process_toplevel(&top, &mut ce, &mut cls_methods, &mut tenv, &mut ass);
        match res {
            Ok(()) => (),
            Err(e) => eprintln!("Error: {:?}", e),
        }
    }
}

fn report_errors<T: Display>(buf: &str, errs: Vec<Rich<T>>) {
    let fname = "REPL";
    for err in errs {
        Report::build(ReportKind::Error, (fname, err.span().into_range()))
            .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
            .with_message(err.reason())
            .with_label(
                Label::new((fname, err.span().into_range()))
                    .with_message(
                        err.found()
                            .map(T::to_string)
                            .unwrap_or_else(|| "end of input".to_string()),
                    )
                    .with_color(Color::Red),
            )
            .with_labels(err.contexts().map(|(l, s)| {
                Label::new((fname, s.into_range()))
                    .with_message(format!("while parsing this {l}"))
                    .with_color(Color::Yellow)
            }))
            .finish()
            .eprint(sources([(fname, buf)]))
            .unwrap();
    }
}

enum TopLevel {
    TypeDef(TypeDef),
    ClassDef(ClassDef),
    ClassImpl(ClassImpl),
    Expr(Expr),
}

struct TypeDef {
    tname: Ustr,
    params: Vec<Ustr>,
    constraints: Vec<(Ustr, Vec<Ustr>)>,
    variants: Vec<VariantDef>,
}

struct VariantDef {
    name: Ustr,
    fields: Vec<TExpr>,
}

struct ClassDef {
    name: Ustr,
    vars: Vec<Ustr>,
    supers: Vec<Ustr>,
    methods: Vec<Declaration>,
}

struct ClassImpl {
    cls: Ustr,
    tys: Vec<TExpr>,
    methods: Vec<Definition>,
}

struct Declaration {
    name: Ustr,
    ty: TExpr,
}

struct Definition {
    name: Ustr,
    alts: Vec<Alt>,
}

#[derive(Debug)]
enum TExpr {
    Sym(Ustr),
    App(Vec<TExpr>),
}

macro_rules! parse_s {
    ($x:expr; let $name:ident = $func:tt => $body:expr) => {{
        $func($x).and_then(|$name| Ok::<_, String>($body))
    }};

    ($x:expr; (let $name:ident = $func:tt) => $body:expr) => {{
        $func($x).and_then(|$name| Ok::<_, String>($body))
    }};

    ($x:expr; $name:ident => $body:expr) => {{
        let $name = $x.clone();
        Ok::<_, String>($body)
    }};

    ($x:expr; _ => $body:expr) => {
        Ok::<_, String>($body)
    };

    ($x:expr; [$($ps:tt)*] => $body:expr) => {
        match $x.as_slice() {
            Some(_xs) => parse_s!(@slice: _xs; [$($ps)*] => $body),
            None => Err("expected list".to_string()),
        }
    };

    ($x:expr; ($fst:tt | $($alts:tt)|+) => $body:expr) => {{
        let res = parse_s!($x; $fst => $body);
        $(
            let res = res.or_else(|_| parse_s!($x; $alts => $body));
        )*
        res
    }};

    ($x:expr; $p:expr => $body:expr) => {
        if $x.eq_symbol($p) {
            Ok::<_, String>($body)
        } else {
            Err(format!("expected {:?}", $p))
        }
    };

    (@slice: $xs:expr; [] => $body:expr) => {
        Ok::<_, String>($body)
    };

    (@slice: $xs:expr; [$p:tt $($ps:tt)*] => $body:expr) => {
        match $xs.get(0) {
            Some(_x) => parse_s!(_x; $p => { parse_s!(@slice: $xs[1..]; [$($ps)*] => $body)? }),
            None => Err("expected more items".to_string()),
        }
    };
}

fn parse_toplevel(sexpr: &S) -> Result<TopLevel> {
    if is_typedef(sexpr) {
        return parse_typedef(sexpr).map(TopLevel::TypeDef);
    }

    if is_classdef(sexpr) {
        return parse_classdef(sexpr).map(TopLevel::ClassDef);
    }

    if is_classimpl(sexpr) {
        return parse_classimpl(sexpr).map(TopLevel::ClassImpl);
    }

    parse_expr(sexpr).map(TopLevel::Expr)
}

fn parse_typedef(sexpr: &S) -> Result<TypeDef> {
    let parts = sexpr.as_slice().ok_or_else(|| "invalid data def")?;
    let ty_field = parts.get(1).ok_or_else(|| "invalid data def")?;

    let tname = parse_typedef_name(ty_field)?;
    let params = parse_typedef_params(ty_field)?;
    let constraints = parse_typedef_constraints(ty_field)?;
    let variants = parse_typedef_variants(&parts[2..])?;

    Ok(TypeDef {
        tname,
        params,
        constraints,
        variants,
    })
}

fn is_typedef(sexpr: &S) -> bool {
    sexpr.as_slice().map_or(false, |x| x[0].eq_symbol("data"))
}

fn parse_typedef_name(ty_field: &S) -> Result<Ustr> {
    parse_s!(ty_field; ([name] | name) => name)
        .ok()
        .as_ref()
        .and_then(S::as_symbol)
        .ok_or_else(|| format!("invalid type name"))
}

fn parse_typedef_params(ty_field: &S) -> Result<Vec<Ustr>> {
    match ty_field.as_slice() {
        None => Ok(vec![]),
        Some(xs) => xs.into_iter().skip(1).map(parse_typedef_param).collect(),
    }
}

fn parse_typedef_param(var_field: &S) -> Result<Ustr> {
    parse_s!(var_field; ([":" name] | name) => name)
        .ok()
        .as_ref()
        .and_then(S::as_symbol)
        .ok_or_else(|| format!("invalid type parameter"))
}

fn parse_typedef_constraints(ty_field: &S) -> Result<Vec<(Ustr, Vec<Ustr>)>> {
    match ty_field.as_slice() {
        None => Ok(vec![]),
        Some(xs) => xs
            .into_iter()
            .skip(1)
            .map(|x| {
                let v = parse_typedef_param(x);
                let cs = parse_typedef_param_constraints(x);
                v.and_then(|v| cs.map(|cs| (v, cs)))
            })
            .collect(),
    }
}

fn parse_typedef_param_constraints(var_field: &S) -> Result<Vec<Ustr>> {
    match var_field {
        S::Symbol(_) => Ok(vec![]),
        S::List(ts) if ts.get(0).map_or(false, |x| x.eq_symbol(":")) => ts
            .into_iter()
            .skip(2)
            .map(|x| {
                x.as_symbol()
                    .ok_or_else(|| format!("invalid constraint: {:?}", x))
            })
            .collect(),

        other => Err(format!("invalid type parameter: {:?}", other)),
    }
}

fn parse_typedef_variants(exprs: &[S]) -> Result<Vec<VariantDef>> {
    exprs.iter().map(parse_typedef_variant).collect()
}

fn parse_typedef_variant(sexpr: &S) -> Result<VariantDef> {
    match sexpr {
        S::Symbol(name) => Ok(VariantDef {
            name: name.clone(),
            fields: vec![],
        }),
        S::List(xs) => {
            let name = xs[0].as_symbol().ok_or_else(|| "invalid variant name")?;
            let fields = xs
                .into_iter()
                .skip(1)
                .map(parse_type_expr)
                .collect::<Result<_>>()?;
            Ok(VariantDef { name, fields })
        }
        _ => Err(format!("invalid variant: {:?}", sexpr)),
    }
}

fn is_classdef(sexpr: &S) -> bool {
    sexpr.as_slice().map_or(false, |x| x[0].eq_symbol("trait"))
}

fn parse_classdef(sexpr: &S) -> Result<ClassDef> {
    Ok(ClassDef {
        name: classdef_name(sexpr)?,
        vars: classdef_vars(sexpr)?,
        supers: vec![],
        methods: parse_classdef_declarations(sexpr)?,
    })
}

fn classdef_name(sexpr: &S) -> Result<Ustr> {
    sexpr
        .as_slice()
        .and_then(|xs| xs.get(1))
        .and_then(S::as_slice)
        .and_then(|xs| xs.get(0))
        .and_then(S::as_symbol)
        .ok_or_else(|| "name must be a symbol".into())
}

fn classdef_vars(sexpr: &S) -> Result<Vec<Ustr>> {
    sexpr
        .as_slice()
        .and_then(|xs| xs.get(1))
        .and_then(S::as_slice)
        .ok_or_else(|| "invalid class".to_string())?
        .into_iter()
        .skip(1)
        .map(S::as_symbol)
        .map(|x| x.ok_or_else(|| "invalid class variable".to_string()))
        .collect()
}

fn parse_classdef_declarations(sexpr: &S) -> Result<Vec<Declaration>> {
    sexpr
        .as_slice()
        .and_then(|xs| xs.get(2..))
        .ok_or_else(|| "invalid class declaration")?
        .into_iter()
        .map(parse_declaration)
        .collect()
}

fn is_classimpl(sexpr: &S) -> bool {
    sexpr.as_slice().map_or(false, |x| x[0].eq_symbol("impl"))
}

fn parse_classimpl(sexpr: &S) -> Result<ClassImpl> {
    match sexpr.as_slice() {
        Some([_, def, methods @ ..]) => match def.as_slice() {
            Some([name, tys @ ..]) => Ok(ClassImpl {
                cls: classimpl_cls(name)?,
                tys: tys.iter().map(parse_type_expr).collect::<Result<_>>()?,
                methods: parse_classimpl_definitions(methods)?,
            }),
            _ => Err("invalid class implementation".into()),
        },
        _ => Err("invalid class implementation".into()),
    }
}

fn classimpl_cls(sexpr: &S) -> Result<Ustr> {
    sexpr.as_symbol().ok_or("name must be a symbol".into())
}

fn parse_classimpl_definitions(sexprs: &[S]) -> Result<Vec<Definition>> {
    sexprs.iter().map(parse_definition).collect()
}

fn parse_declaration(sexpr: &S) -> Result<Declaration> {
    match sexpr.as_slice() {
        Some([prefix, name, tx]) => {
            if !prefix.eq_symbol(":") {
                return Err("invalid declaration".into());
            }
            let name = name.as_symbol().ok_or("name must be a symbol")?;
            let ty = parse_type_expr(tx)?;
            Ok(Declaration { name, ty })
        }
        _ => Err("invalid class declaration".into()),
    }
}

fn parse_definition(sexpr: &S) -> Result<Definition> {
    match sexpr.as_slice() {
        Some([name, arms @ ..]) => {
            let name = name.as_symbol().ok_or("name must be a symbol")?;
            let alts = parse_alts(arms)?;
            Ok(Definition { name, alts })
        }
        _ => Err("invalid function definition".into()),
    }
}

fn parse_type_expr(sexpr: &S) -> Result<TExpr> {
    match sexpr {
        S::Symbol(name) => Ok(TExpr::Sym(*name)),
        S::List(xs) => xs
            .into_iter()
            .map(parse_type_expr)
            .collect::<Result<_>>()
            .map(TExpr::App),
        _ => Err(format!("invalid type expression: {:?}", sexpr)),
    }
}

fn parse_symbol(sexpr: &S) -> Result<Ustr> {
    match sexpr {
        S::Symbol(name) => Ok(*name),
        _ => Err(format!("expected symbol: {:?}", sexpr)),
    }
}

fn process_toplevel(
    tl: &Spanned<ast::TopLevel>,
    ce: &mut ClassEnv,
    cls_methods: &mut im_rc::HashMap<Ustr, Vec<Ustr>>,
    tenv: &mut im_rc::HashMap<Ustr, Type>,
    ass: &mut Vec<Assump>,
) -> Result<()> {
    match &tl.inner {
        ast::TopLevel::TypeDef(def) => {
            let (ty, ass_) = process_typedef(&def, &tenv)?;
            tenv.insert(def.tname.inner.0, ty);
            ass.extend(ass_);
        }
        ast::TopLevel::ClassDef(def) => {
            let supers = def.supers.iter().map(|c| ustr(&c.cls.0)).collect();
            let ce_ = EnvTransformer::add_class(def.inner.cname.0, supers).apply(&ce)?;

            let mut cls_tenv = tenv.clone();

            let tvs: Vec<_> = def.params.iter().map(|v| Tyvar(v.0, Kind::Star)).collect();
            for (v, tv) in def.params.iter().zip(&tvs) {
                let tvar = Type::TVar(tv.clone());
                cls_tenv.insert(v.0, tvar.clone());
            }

            if tvs.len() != 1 {
                return Err("class must have exactly one type variable".into());
            }

            let tvar = Type::TVar(tvs[0].clone());

            for decl in &def.methods {
                let ty = process_type_expr(&decl.ty, &cls_tenv)?;
                let sc = Scheme::quantify_by_var_order(
                    &tvs,
                    &Qual(vec![Pred::IsIn(def.cname.0, tvar.clone())], ty),
                );

                // todo: not sure if I want class methods to be global functions
                ass.push(Assump { i: decl.name.0, sc });

                cls_methods
                    .entry(def.cname.0)
                    .or_default()
                    .push(decl.name.0);
            }

            *ce = ce_;
        }

        /*TopLevel::ClassImpl(impl_) => {
        if impl_.tys.len() != 1 {
        return Err("classes support exactly one type parameter".into());
        }

        let ty = process_type_expr(&impl_.tys[0], &tenv)?;

        let ce_ =
        EnvTransformer::add_inst(vec![], Pred::IsIn(impl_.cls, ty.clone())).apply(&ce)?;

        let mut required_methods: HashSet<_> = cls_methods
        .get(&impl_.cls)
        .into_iter()
        .flatten()
        .copied()
        .collect();

        for Definition { name: mname, alts } in &impl_.methods {
        if !required_methods.remove(mname) {
        return Err(format!("Unexpected method: {}", mname));
        }

        let Assump { sc, .. } = ass
        .iter()
        .rev()
        .find(|&a| a.i == *mname)
        .ok_or("method in assumptions")?;

        let mut ti = TI::new();

        let sc = dbg!(sc.inst(&[ty.clone()]));
        let expl = Expl(*mname, sc.clone(), alts.clone());

        let ps = ti_expl(&mut ti, &ce_, &ass, &expl)?;
        println!("Inferred: {:?}, {:?}", ps, ti);
        }

        if !required_methods.is_empty() {
        return Err(format!("Missing methods: {:?}", required_methods));
        }

        *ce = ce_;
        }*/
        ast::TopLevel::Expr(xp) => {
            let mut ti = TI::new();

            match ti_expr(&mut ti, &ce, &ass, &convert_expression(xp))
                .and_then(|(ps, t)| finalize(ti, &ce, &ps, &t))
            {
                Ok(t) => {
                    println!("Inferred: {:?}", t);
                }
                Err(e) => return Err(e),
            }
        }
    }

    Ok(())
}

fn process_typedef(
    td: &Spanned<ast::TypeDef>,
    tenv: &im_rc::HashMap<Ustr, Type>,
) -> Result<(Type, Vec<Assump>)> {
    // assuming all parameters are of kind *
    let vars: Vec<_> = td
        .params
        .iter()
        .map(|p| Tyvar(p.inner.0.clone(), Kind::Star))
        .collect();

    let mut kind = Kind::Star;
    for Tyvar(_, k) in &vars {
        kind = Kind::kfun(k.clone(), kind);
    }

    let tcons = Type::TCon(Tycon(td.tname.inner.0, kind));

    let mut dtype = tcons.clone();
    for v in vars.iter().cloned() {
        dtype = Type::tapp(dtype, Type::TVar(v));
    }

    let mut preds = vec![];
    for Spanned {
        inner: ast::Constraint { cls, tys },
        ..
    } in &td.constraints
    {
        for t in tys {
            let v = match &t.inner {
                ast::TExpr::Sym(s) => s,
                _ => unimplemented!(),
            };
            let tv = vars
                .iter()
                .find(|Tyvar(name, _)| name == v)
                .ok_or_else(|| format!("unbound type variable: {}", v))?
                .clone();
            preds.push(Pred::IsIn(cls.inner.0.clone(), Type::TVar(tv.clone())));
            preds.push(Pred::IsIn(cls.inner.0.clone(), Type::TVar(tv)));
        }
    }

    let mut local_tenv = tenv.clone();

    // allow recursive types
    local_tenv.insert(td.tname.inner.0, tcons.clone());

    for v in vars.iter().cloned() {
        local_tenv.insert(v.0.clone(), Type::TVar(v));
    }

    let mut ass = vec![];
    for variant in &td.variants {
        let mut vtype = dtype.clone();

        for field in &variant.fields {
            let field_type = process_type_expr(field, &local_tenv)?;
            vtype = Type::func(field_type, vtype);
        }

        let sc = Scheme::quantify_by_var_order(&vars, &Qual(preds.clone(), vtype));
        ass.push(Assump {
            i: variant.name.inner.0,
            sc,
        })
    }

    Ok((tcons, ass))
}

fn process_type_expr(tx: &Spanned<ast::TExpr>, tenv: &im_rc::HashMap<Ustr, Type>) -> Result<Type> {
    match &tx.inner {
        ast::TExpr::Sym(name) => tenv
            .get(&name)
            .cloned()
            .ok_or_else(|| format!("unknown type: {}", name)),

        ast::TExpr::App(lhs, rhs) => {
            let f = process_type_expr(lhs, tenv)?;
            let a = process_type_expr(rhs, tenv)?;
            Ok(Type::tapp(f, a))
        }
    }
}

fn classdef_functions(sexpr: &S, tenv: &im_rc::HashMap<Ustr, Type>) -> Result<Vec<(Ustr, Type)>> {
    let mut fns = vec![];
    for decl in &sexpr.as_slice().unwrap()[2..] {
        match decl.as_slice() {
            Some([name, ty]) => {
                let name = name.as_symbol().ok_or("name must be a symbol")?;
                let ty = parse_type(ty, tenv)?;
                fns.push((name, ty));
            }
            _ => return Err("invalid class declaration".into()),
        }
    }

    Ok(fns)
}

fn classimpl_ty(sexpr: &S, tenv: &im_rc::HashMap<Ustr, Type>) -> Result<Type> {
    let ty = &sexpr.as_slice().unwrap()[2];
    parse_type(ty, tenv)
}

fn classimpl_defs(sexpr: &S) -> Result<Vec<(Ustr, Vec<Alt>)>> {
    let mut defs = vec![];
    for x in &sexpr.as_slice().unwrap()[3..] {
        match x.as_slice() {
            Some([name, arms @ ..]) => {
                let name = name.as_symbol().ok_or("name must be a symbol")?;
                defs.push((name, parse_alts(arms)?))
            }
            _ => return Err("invalid function definition".into()),
        }
    }
    Ok(defs)
}

fn parse_alts(arms: &[S]) -> Result<Vec<Alt>> {
    arms.iter().map(parse_alt).collect()
}

fn parse_alt(arm: &S) -> Result<Alt> {
    match arm.as_slice() {
        Some([pats, body]) => {
            let pat = pats
                .as_slice()
                .ok_or("invalid patterns")?
                .into_iter()
                .map(parse_pat)
                .collect::<Result<_>>()?;
            let body = parse_expr(body)?;
            Ok(Alt(pat, body))
        }
        _ => Err(format!("invalid arm {arm:?}")),
    }
}

fn parse_pat(pat: &S) -> Result<Pat> {
    match pat {
        p if p.eq_symbol("_") => Ok(Pat::PWildcard),
        S::Symbol(sym) => Ok(Pat::PVar(*sym)),
        _ => Err("invalid pattern".into()),
    }
}

fn parse_type(sexpr: &S, tenv: &im_rc::HashMap<Ustr, Type>) -> Result<Type> {
    if let Some(xs) = sexpr.as_slice() {
        let mut t = parse_type(&xs[0], tenv)?;
        for x in xs[1..].iter() {
            t = Type::tapp(t, parse_type(x, tenv)?);
        }
        return Ok(t);
    }

    match sexpr.as_symbol() {
        Some(name) => tenv
            .get(&name)
            .cloned()
            .ok_or_else(|| format!("unknown type: {}", name)),
        None => Err("invalid type".into()),
    }
}

fn parse_expr(sexpr: &S) -> Result<Expr> {
    match sexpr {
        S::Int(i) => Ok(Expr::Lit(Literal::Int(*i))),
        S::Float(f) => Ok(Expr::Lit(Literal::Rat(*f))),
        S::Symbol(sym) => Ok(Expr::Var(*sym)),

        S::List(xs) if xs.get(0).map_or(false, |x| x.eq_symbol("quote")) => match xs[1] {
            S::Symbol(sym) => Ok(Expr::Lit(Literal::Sym(sym))),
            _ => todo!("{:?}", sexpr),
        },

        S::List(xs) if xs.len() > 0 => {
            let mut xs_ = xs.iter().map(|x| parse_expr(x));
            let mut x = xs_.next().unwrap()?;
            for a in xs_ {
                x = Expr::App(Rc::new(x), Rc::new(a?));
            }
            Ok(x)
        }
        _ => todo!("{:?}", sexpr),
    }
}

type Int = usize;
type Id = Ustr;

#[derive(Clone, Debug, PartialEq)]
enum S {
    Int(i64),
    Float(f64),
    Symbol(Ustr),
    String(Ustr),
    List(Vec<S>),
}

impl S {
    fn eq_symbol(&self, s: &str) -> bool {
        matches!(self, S::Symbol(x) if x == s)
    }

    fn as_symbol(&self) -> Option<Ustr> {
        match self {
            S::Symbol(x) => Some(*x),
            _ => None,
        }
    }

    fn is_list(&self) -> bool {
        matches!(self, S::List(_))
    }

    fn as_slice(&self) -> Option<&[S]> {
        match self {
            S::List(x) => Some(x),
            _ => None,
        }
    }
}

struct SF;

impl SexprFactory for SF {
    type Sexpr = S;
    type Integer = i64;
    type Float = f64;

    fn int(&mut self, x: Self::Integer) -> Self::Sexpr {
        S::Int(x)
    }

    fn float(&mut self, x: Self::Float) -> Self::Sexpr {
        S::Float(x)
    }

    fn symbol(&mut self, x: &str) -> Self::Sexpr {
        S::Symbol(Ustr::from(x))
    }

    fn string(&mut self, x: &str) -> Self::Sexpr {
        S::String(Ustr::from(x))
    }

    fn list(&mut self, x: Vec<Self::Sexpr>) -> Self::Sexpr {
        S::List(x)
    }

    fn pair(&mut self, a: Self::Sexpr, b: Self::Sexpr) -> Self::Sexpr {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_binding() {
        let s = S::Int(42);
        assert_eq!(parse_s!(s; x => x), Ok::<S, String>(S::Int(42)));
    }

    #[test]
    fn parse_list() {
        assert!(parse_s!(S::Int(42); [] => 0).is_err());
        assert_eq!(parse_s!(S::List(vec![]); [] => 0), Ok(0));

        assert!((|| parse_s!(S::List(vec![]); [_] => 0))().is_err(),);

        assert_eq!(
            (|| parse_s!(S::List(vec![S::Int(1), S::Int(2), S::Int(3)]); [x y x] => (x, y)))(),
            Ok::<_, String>((S::Int(3), S::Int(2)))
        );
    }

    #[test]
    fn parse_literal() {
        assert_eq!(
            parse_s!(S::Symbol(ustr("foo")); "foo" => ()),
            Ok::<_, String>(())
        );
        assert!(parse_s!(S::Int(42); "foo" => ()).is_err());
        assert!(parse_s!(S::Symbol(ustr("bar")); "foo" => ()).is_err());
    }

    #[test]
    fn parse_function() {
        assert_eq!(
            parse_s!(S::Int(42); let x = (|_| Ok(0)) => x),
            Ok::<_, String>(0)
        );
    }
}
