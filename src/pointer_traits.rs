use std::{convert::TryInto, ptr::NonNull};

/// TensorBaseのptrに当たる構造体が必ずimplされなければならないトレイト
pub trait TensorPointer: Sized {
    type Elem;
    /// Check if the offset value is in the allocated memory area
    fn is_inbound(&self, offset: isize) -> bool {
        self.len() > offset.try_into().unwrap()
    }
    /// clone mem and make vec
    fn to_vec(&self) -> Vec<Self::Elem>;
    /// Create Self from a vector
    fn from_vec(vec: Vec<Self::Elem>) -> Self;
    /// Pointerで保持されている先頭から、offset分だけoffsetしNonNUllを返す
    fn offset(&self, offset: isize) -> NonNull<Self::Elem>;
    /// Pointerで保持されている先頭のポインタを返す
    fn as_ptr(&self) -> *const Self::Elem;
    /// 確保されているポインタの先頭からみて何個の要素が確保されているかを返す
    fn len(&self) -> usize;
    /// retun offset
    fn offset_num(&self) -> usize;
}

/// データに関してDropする責任があるポインタ
pub trait Owned: TensorPointer {
    type View;
    type ViewMut;
    /// 確保したメモリをそのままクローンする。
    /// (shapeやstrideを考慮せず、メモリのレイアウトそのままクローンする)
    fn clone_mem_layout(&self) -> Self {
        let v = self.to_vec();
        Self::from_vec(v)
    }
    /// pointerをviewにキャストする
    fn to_view(&self, offset: usize) -> Self::View;
    /// pointerをview_mutにキャストする
    fn to_view_mut(&mut self, offset: usize) -> Self::ViewMut;
}

/// データに関して、可変であるポインタ
pub trait Mut: TensorPointer {
    fn assign_region<P>(&mut self, other: &P, offset: usize, region: usize)
    where
        P: TensorPointer<Elem = <Self as TensorPointer>::Elem>;
}

/// データの参照を持つポインタ
pub trait View<V: View<V, O>, O: Owned>: TensorPointer {
    // Return がSelfではない理由
    // 返り値がSelfではない理由はViewMutに実装した際にViewMutではなくViewが返り値になってほしいから
    /// offsetで指定された場所を起点にして、regionで指定された長さを持つViewのポインタを返す
    fn access_by_offset_region(&self, offset: usize, region: usize) -> V;

    /// Cast for Pointer wihch impl Owned
    fn to_owned(&self) -> O;
}

pub trait ViewMut<V: View<V, O>, O: Owned>: View<V, O> + Mut {}

/// Impl to Cpu Pointer only
pub trait Cpu: TensorPointer {
    fn to_slice(&'_ self) -> &'_ [<Self as TensorPointer>::Elem];
}
