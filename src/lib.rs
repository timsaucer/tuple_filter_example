use arrow::buffer::BooleanBuffer;
use arrow::datatypes::DataType;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow::{
    array::{Array, ArrayData, BooleanArray, PrimitiveArray, StringArray},
    compute,
    datatypes::Int64Type,
};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct TupleFilterClass {
    values_of_interest: Vec<(i64, i64, String)>,
}

#[pymethods]
impl TupleFilterClass {
    #[new]
    fn new(values_of_interest: Vec<(i64, i64, String)>) -> Self {
        Self { values_of_interest }
    }

    fn __call__(
        &self,
        py: Python<'_>,
        partkey_expr: &Bound<'_, PyAny>,
        suppkey_expr: &Bound<'_, PyAny>,
        returnflag_expr: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let partkey_arr: PrimitiveArray<Int64Type> =
            ArrayData::from_pyarrow_bound(partkey_expr)?.into();
        let suppkey_arr: PrimitiveArray<Int64Type> =
            ArrayData::from_pyarrow_bound(suppkey_expr)?.into();
        let returnflag_arr: StringArray = ArrayData::from_pyarrow_bound(returnflag_expr)?.into();

        let mut res: Option<BooleanArray> = None;

        for (partkey, suppkey, returnflag) in &self.values_of_interest {
            let filtered_partkey_arr = BooleanArray::from_unary(&partkey_arr, |p| p == *partkey);
            let filtered_suppkey_arr = BooleanArray::from_unary(&suppkey_arr, |s| s == *suppkey);
            let filtered_returnflag_arr =
                BooleanArray::from_unary(&returnflag_arr, |s| s == returnflag);

            let part_and_supp = compute::and(&filtered_partkey_arr, &filtered_suppkey_arr)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let resultant_arr = compute::and(&part_and_supp, &filtered_returnflag_arr)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            res = match res {
                Some(r) => compute::or(&r, &resultant_arr).ok(),
                None => Some(resultant_arr),
            };
        }

        res.unwrap().into_data().to_pyarrow(py)
    }
}

#[pyclass]
pub struct TupleFilterDirectIterationClass {
    values_of_interest: Vec<(i64, i64, String)>,
}

#[pymethods]
impl TupleFilterDirectIterationClass {
    #[new]
    fn new(values_of_interest: Vec<(i64, i64, String)>) -> Self {
        Self { values_of_interest }
    }

    fn __call__(
        &self,
        py: Python<'_>,
        partkey_expr: &Bound<'_, PyAny>,
        suppkey_expr: &Bound<'_, PyAny>,
        returnflag_expr: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let partkey_arr: PrimitiveArray<Int64Type> =
            ArrayData::from_pyarrow_bound(partkey_expr)?.into();
        let suppkey_arr: PrimitiveArray<Int64Type> =
            ArrayData::from_pyarrow_bound(suppkey_expr)?.into();
        let returnflag_arr: StringArray = ArrayData::from_pyarrow_bound(returnflag_expr)?.into();

        if partkey_arr.len() != suppkey_arr.len() || partkey_arr.len() != returnflag_arr.len() {
            return Err(PyValueError::new_err(
                "Cannot perform tuple filter on arrays of different length".to_string(),
            ));
        }

        if partkey_arr.is_empty() {
            return BooleanArray::from(ArrayData::new_empty(&DataType::Boolean))
                .into_data()
                .to_pyarrow(py);
        }

        let values_to_search: Vec<(&i64, &i64, &str)> = (&self.values_of_interest)
            .iter()
            .map(|(a, b, c)| (a, b, c.as_str()))
            .collect();

        let values = partkey_arr
            .values()
            .iter()
            .zip(suppkey_arr.values().iter())
            .zip(returnflag_arr.iter())
            .map(|((a, b), c)| (a, b, c))
            .map(|(a, b, c)| values_to_search.contains(&(a, b, c.unwrap_or_default())));

        let res: BooleanArray = BooleanBuffer::from_iter(values).into();

        res.into_data().to_pyarrow(py)
    }
}

#[pyfunction]
pub fn tuple_filter_fn(
    py: Python<'_>,
    partkey_expr: &Bound<'_, PyAny>,
    suppkey_expr: &Bound<'_, PyAny>,
    returnflag_expr: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let partkey_arr: PrimitiveArray<Int64Type> =
        ArrayData::from_pyarrow_bound(partkey_expr)?.into();
    let suppkey_arr: PrimitiveArray<Int64Type> =
        ArrayData::from_pyarrow_bound(suppkey_expr)?.into();
    let returnflag_arr: StringArray = ArrayData::from_pyarrow_bound(returnflag_expr)?.into();

    let values_of_interest = vec![
        (1530, 4031, "N".to_string()),
        (6530, 1531, "N".to_string()),
        (5618, 619, "N".to_string()),
        (8118, 8119, "N".to_string()),
    ];

    let mut res: Option<BooleanArray> = None;

    for (partkey, suppkey, returnflag) in &values_of_interest {
        let filtered_partkey_arr = BooleanArray::from_unary(&partkey_arr, |p| p == *partkey);
        let filtered_suppkey_arr = BooleanArray::from_unary(&suppkey_arr, |s| s == *suppkey);
        let filtered_returnflag_arr =
            BooleanArray::from_unary(&returnflag_arr, |s| s == returnflag);

        let part_and_supp = compute::and(&filtered_partkey_arr, &filtered_suppkey_arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let resultant_arr = compute::and(&part_and_supp, &filtered_returnflag_arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        res = match res {
            Some(r) => compute::or(&r, &resultant_arr).ok(),
            None => Some(resultant_arr),
        };
    }

    res.unwrap().into_data().to_pyarrow(py)
}

#[pymodule]
fn tuple_filter_example(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(tuple_filter_fn, module)?)?;
    module.add_class::<TupleFilterClass>()?;
    module.add_class::<TupleFilterDirectIterationClass>()?;
    Ok(())
}
