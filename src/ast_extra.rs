use crate::assumptions::Assump;
use crate::kinds::Kind;
use crate::lists::List;
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::BindGroup;
use crate::types::Type;
use crate::Id;
use std::collections::HashMap;

#[derive(Debug)]
pub enum TopLevel {
    DefClass(DefClass),
    ImplClass(ImplClass),
    BindGroup(BindGroup),
}

#[derive(Debug)]
pub struct DefClass {
    pub name: Id,
    pub super_classes: Vec<Id>,
    pub methods: Vec<Assump>,
}

#[derive(Debug)]
pub struct ImplClass {
    pub cls: Id,
    pub ty: Id,
    pub methods: BindGroup,
}

#[derive(Debug)]
pub struct SchemeBuilder {
    genvars: Vec<(Id, Kind, Vec<Id>)>,
    ty: Type,
}

impl SchemeBuilder {
    pub fn new(genvars: Vec<(Id, Kind, Vec<Id>)>, ty: Type) -> Self {
        SchemeBuilder { genvars, ty }
    }
    pub fn build(self) -> Scheme {
        let mut vars = HashMap::new();
        let mut kinds = List::Nil;
        let mut preds = vec![];
        for (name, kind, constraints) in self.genvars {
            let idx = vars.len();
            vars.insert(name, (Type::TGen(idx), kind.clone()));
            kinds = kinds.cons(kind);

            for c in constraints {
                let pred = Pred::IsIn(c, Type::TGen(idx));
                preds.push(pred);
            }
        }

        let ty_ = self.ty.subst(&vars);

        let qual_ty = Qual(preds, ty_);
        Scheme::Forall(kinds, qual_ty)
    }
}
