use std::ops::Range;

use crate::error::NeuralNetworkError;

#[derive(Eq, PartialEq, Debug, Clone, Default, PartialOrd, Ord, Hash)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides: Vec<usize> = Self::compute_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn with_offset(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let dimension: usize = shape.len();
        let mut strides: Vec<usize> = vec![1; dimension];
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

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_contiguous(&self) -> bool {
        let expected_strides: Vec<usize> = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    pub fn permute(&self, dimensions: &[usize]) -> Result<Self, NeuralNetworkError> {
        if dimensions.len() != self.ndim() {
            return Err(NeuralNetworkError::InvalidDimension {
                got: dimensions.len(),
                max_dimension: self.ndim(),
            });
        }
        let mut seen: Vec<bool> = vec![false; self.ndim()];
        for &dimension in dimensions {
            if dimension >= self.ndim() || seen[dimension] {
                return Err(NeuralNetworkError::InvalidDimension {
                    got: dimensions.len(),
                    max_dimension: self.ndim(),
                });
            }
            seen[dimension] = true;
        }
        let new_shape: Vec<usize> = dimensions
            .iter()
            .map(|&dimension: &usize| self.shape[dimension])
            .collect();
        let new_strides: Vec<usize> = dimensions
            .iter()
            .map(|&dimension: &usize| self.strides[dimension])
            .collect();
        Ok(Layout {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuralNetworkError> {
        if !self.is_contiguous() {
            return Err(NeuralNetworkError::NotContiguous);
        }
        let old_size: usize = self.num_elements();
        let new_size: usize = new_shape.iter().product();
        if old_size != new_size {
            return Err(NeuralNetworkError::IncompatibleShape {
                shape_given: new_shape,
                tensor_shape: self.shape().to_vec(),
            });
        }
        Ok(Layout::new(new_shape))
    }

    pub fn slice(&self, dimension: usize, range: Range<usize>) -> Result<Self, NeuralNetworkError> {
        if dimension >= self.ndim() {
            return Err(NeuralNetworkError::InvalidDimension {
                got: dimension,
                max_dimension: self.ndim(),
            });
        }
        if range.end > self.shape[dimension] {
            return Err(NeuralNetworkError::OutOfBounds);
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape[dimension] = range.len();
        let new_offset: usize = self.offset + range.start * self.strides[dimension];
        Ok(Layout {
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let shape: Vec<usize> = vec![2, 3, 4];
        let layout: Layout = Layout::new(shape);
        assert_eq!(layout.shape, vec![2, 3, 4]);
        assert_eq!(layout.strides, vec![12, 4, 1]);
        assert_eq!(layout.offset, 0);
    }

    #[test]
    fn test_new_1d() {
        let shape: Vec<usize> = vec![5];
        let layout: Layout = Layout::new(shape);
        assert_eq!(layout.shape, vec![5]);
        assert_eq!(layout.strides, vec![1]);
        assert_eq!(layout.offset, 0);
    }

    #[test]
    fn test_new_empty() {
        let shape: Vec<usize> = vec![];
        let layout: Layout = Layout::new(shape);
        assert_eq!(layout.shape, Vec::<usize>::new());
        assert_eq!(layout.strides, Vec::<usize>::new());
        assert_eq!(layout.offset, 0);
    }

    #[test]
    fn test_with_offset() {
        let shape: Vec<usize> = vec![2, 3];
        let strides: Vec<usize> = vec![3, 1];
        let offset: usize = 5;
        let layout: Layout = Layout::with_offset(shape, strides, offset);
        assert_eq!(layout.shape, vec![2, 3]);
        assert_eq!(layout.strides, vec![3, 1]);
        assert_eq!(layout.offset, 5);
    }

    #[test]
    fn test_shape() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        assert_eq!(layout.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_strides() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        assert_eq!(layout.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_offset() {
        let layout: Layout = Layout::with_offset(vec![2, 3], vec![3, 1], 10);
        assert_eq!(layout.offset(), 10);
    }

    #[test]
    fn test_ndim() {
        let layout1: Layout = Layout::new(vec![2, 3, 4]);
        let layout2: Layout = Layout::new(vec![5]);
        let layout3: Layout = Layout::new(vec![]);
        assert_eq!(layout1.ndim(), 3);
        assert_eq!(layout2.ndim(), 1);
        assert_eq!(layout3.ndim(), 0);
    }

    #[test]
    fn test_num_elements() {
        let layout1: Layout = Layout::new(vec![2, 3, 4]);
        let layout2: Layout = Layout::new(vec![5]);
        let layout3: Layout = Layout::new(vec![]);
        assert_eq!(layout1.num_elements(), 24);
        assert_eq!(layout2.num_elements(), 5);
        assert_eq!(layout3.num_elements(), 1);
    }

    #[test]
    fn test_index() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        assert_eq!(layout.index(&[0, 0, 0]), 0);
        assert_eq!(layout.index(&[0, 0, 1]), 1);
        assert_eq!(layout.index(&[0, 1, 0]), 4);
        assert_eq!(layout.index(&[1, 0, 0]), 12);
        assert_eq!(layout.index(&[1, 2, 3]), 23);
    }

    #[test]
    fn test_index_with_offset() {
        let layout: Layout = Layout::with_offset(vec![2, 3], vec![3, 1], 10);
        assert_eq!(layout.index(&[0, 0]), 10);
        assert_eq!(layout.index(&[0, 1]), 11);
        assert_eq!(layout.index(&[1, 0]), 13);
        assert_eq!(layout.index(&[1, 2]), 15);
    }

    #[test]
    fn test_is_contiguous() {
        let contiguous: Layout = Layout::new(vec![2, 3, 4]);
        assert!(contiguous.is_contiguous());
    }

    #[test]
    fn test_is_not_contiguous() {
        let non_contiguous: Layout = Layout::with_offset(vec![2, 3], vec![1, 3], 0);
        assert!(!non_contiguous.is_contiguous());
    }

    #[test]
    fn test_permute() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        let permuted: Result<Layout, NeuralNetworkError> = layout.permute(&[2, 0, 1]);
        assert!(permuted.is_ok());
        let permuted: Layout = permuted.unwrap();
        assert_eq!(permuted.shape(), &[4, 2, 3]);
        assert_eq!(permuted.strides(), &[1, 12, 4]);
        assert_eq!(permuted.offset(), 0);
    }

    #[test]
    fn test_permute_invalid_dimension() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        let result: Result<Layout, NeuralNetworkError> = layout.permute(&[0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_permute_out_of_bounds() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        let result: Result<Layout, NeuralNetworkError> = layout.permute(&[0, 1, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_permute_duplicate() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        let result: Result<Layout, NeuralNetworkError> = layout.permute(&[0, 1, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        let reshaped: Result<Layout, NeuralNetworkError> = layout.reshape(vec![6, 4]);
        assert!(reshaped.is_ok());
        let reshaped: Layout = reshaped.unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(reshaped.num_elements(), 24);
    }

    #[test]
    fn test_reshape_incompatible_size() {
        let layout: Layout = Layout::new(vec![2, 3, 4]);
        let result: Result<Layout, NeuralNetworkError> = layout.reshape(vec![5, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_non_contiguous() {
        let non_contiguous: Layout = Layout::with_offset(vec![2, 3], vec![1, 3], 0);
        let result: Result<Layout, NeuralNetworkError> = non_contiguous.reshape(vec![6]);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice() {
        let layout: Layout = Layout::new(vec![5, 4, 3]);
        let sliced: Result<Layout, NeuralNetworkError> = layout.slice(0, 1..3);
        assert!(sliced.is_ok());
        let sliced: Layout = sliced.unwrap();
        assert_eq!(sliced.shape(), &[2, 4, 3]);
        assert_eq!(sliced.strides(), &[12, 3, 1]);
        assert_eq!(sliced.offset(), 12);
    }

    #[test]
    fn test_slice_invalid_dimension() {
        let layout: Layout = Layout::new(vec![5, 4, 3]);
        let result: Result<Layout, NeuralNetworkError> = layout.slice(3, 0..2);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let layout: Layout = Layout::new(vec![5, 4, 3]);
        let result: Result<Layout, NeuralNetworkError> = layout.slice(0, 0..10);
        assert!(result.is_err());
    }

    #[test]
    fn test_default() {
        let layout: Layout = Layout::default();
        assert_eq!(layout.shape(), &[]);
        assert_eq!(layout.strides(), &[]);
        assert_eq!(layout.offset(), 0);
    }

    #[test]
    fn test_clone() {
        let original: Layout = Layout::new(vec![2, 3, 4]);
        let cloned: Layout = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_partial_eq() {
        let layout1: Layout = Layout::new(vec![2, 3]);
        let layout2: Layout = Layout::new(vec![2, 3]);
        let layout3: Layout = Layout::new(vec![3, 2]);
        assert_eq!(layout1, layout2);
        assert_ne!(layout1, layout3);
    }
}
