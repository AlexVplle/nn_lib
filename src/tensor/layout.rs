#[derive(Eq, PartialEq, Debug, Clone, Copy, Default, PartialOrd, Ord, Hash)]
struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let dimension: usize = shape.len();
        let mut strides: Vec<usize> = vec![1, dimension];
        for i in (0..dimension.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());

        self.offset
            + indices
                .iter()
                .zip(&self.strides)
                .map(|(index, stride): (&usize, &usize)| index * stride)
                .sum::<usize>()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_contiguous(&self) -> bool {}
}
