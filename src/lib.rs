use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow::{
    array::{Array, ArrayData, BooleanArray, PrimitiveArray, StringArray},
    compute,
    datatypes::Int32Type,
};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
pub fn tuple_filter_fn(
    py: Python<'_>,
    partkey_expr: &Bound<'_, PyAny>,
    suppkey_expr: &Bound<'_, PyAny>,
    returnflag_expr: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let partkey_arr: PrimitiveArray<Int32Type> =
        ArrayData::from_pyarrow_bound(partkey_expr)?.into();
    let suppkey_arr: PrimitiveArray<Int32Type> =
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
    module.add_function(wrap_pyfunction!(tuple_filter_fn, module)?)
}
