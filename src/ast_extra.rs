use crate::assumptions::Assump;
use crate::specific_inference::BindGroup;
use crate::Id;

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
