use std::ptr::NonNull;

pub trait TensorPointer: Sized {
    type Elem;
    fn is_inbound(&self, offset: isize) -> bool;
    /// clone mem and make vec
    fn to_vec(&self) -> Vec<Self::Elem>;
    fn from_vec(vec: Vec<Self::Elem>) -> Self;
    /// Pointerで保持されている先頭から、offset分だけoffsetしNonNUllを返す
    fn offset(&self, offset: isize) -> NonNull<Self::Elem>;
    /// Pointerで保持されている先頭のポインタを返す
    fn as_ptr(&self) -> *const Self::Elem;
    fn len(&self) -> usize;
    fn offset_num(&self) -> usize;
}

pub trait Owned: TensorPointer {
    type View;
    type ViewMut;
    fn clone_mem_layout(&self) -> Self {
        let v = self.to_vec();
        Self::from_vec(v)
    }

    fn to_view(&self, offset: usize) -> Self::View;
    fn to_view_mut(&mut self, offset: usize) -> Self::ViewMut;
}

pub trait Mut: TensorPointer {
    fn assign_region<P>(&mut self, other: &P, offset: usize, region: usize)
    where
        P: TensorPointer<Elem = <Self as TensorPointer>::Elem>;
}

pub trait View: TensorPointer {
    type AccessOutput: View;
    type OwnedOutput: Owned;

    fn access_by_offset_region(&self, offset: usize, region: usize) -> Self::AccessOutput;

    fn to_owned(&self) -> Self::OwnedOutput;
}

pub trait ViewMut: View + Mut {}
