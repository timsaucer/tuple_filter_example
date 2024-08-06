# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import List
from datafusion import SessionContext, col, lit, udf, functions as F
import os
from datafusion.expr import Expr
import pyarrow as pa
import pyarrow.compute as pc
import time

from tuple_filter_example import tuple_filter_fn, TupleFilterClass, TupleFilterDirectIterationClass

path = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(path, "./lineitem.parquet")

# This example serves to demonstrate alternate approaches to answering the
# question "return all of the rows that have a specific combination of these
# values". We have the combinations we care about provided as a python
# list of tuples. There is no built in function that supports this operation,
# but it can be explicilty specified via a single expression or we can
# use a user defined function.

ctx = SessionContext()

# These part keys and suppliers are chosen because there are
# cases where two suppliers each have two of the part keys
# but we are interested in these specific combinations.

values_of_interest = [
    (1530, 4031, "N"),
    (6530, 1531, "N"),
    (5618, 619, "N"),
    (8118, 8119, "N"),
]

partkeys = [lit(r[0]) for r in values_of_interest]
suppkeys = [lit(r[1]) for r in values_of_interest]
returnflags = [lit(r[2]) for r in values_of_interest]

df_lineitem = ctx.read_parquet(filepath).select(
    col("l_partkey"), col("l_suppkey"), col("l_returnflag")
)

start_time = time.time()

df_simple_filter = (df_lineitem
    .filter(F.in_list(col("l_partkey"), partkeys, negated=False))
    .filter(F.in_list(col("l_suppkey"), suppkeys, negated=False))
    .filter(F.in_list(col("l_returnflag"), returnflags, negated=False))
)

num_rows = df_simple_filter.count()
print(
    f"Simple filtering has number {num_rows} rows and took {time.time() - start_time} s"
)
print("This is the incorrect number of rows!")
start_time = time.time()

# Explicitly check for the combinations of interest.
# This works but is not scalable.

filter_expr = (
    (
        (col("l_partkey") == lit(values_of_interest[0][0]))
        & (col("l_suppkey") == lit(values_of_interest[0][1]))
        & (col("l_returnflag") == lit(values_of_interest[0][2]))
    )
    | (
        (col("l_partkey") == lit(values_of_interest[1][0]))
        & (col("l_suppkey") == lit(values_of_interest[1][1]))
        & (col("l_returnflag") == lit(values_of_interest[1][2]))
    )
    | (
        (col("l_partkey") == lit(values_of_interest[2][0]))
        & (col("l_suppkey") == lit(values_of_interest[2][1]))
        & (col("l_returnflag") == lit(values_of_interest[2][2]))
    )
    | (
        (col("l_partkey") == lit(values_of_interest[3][0]))
        & (col("l_suppkey") == lit(values_of_interest[3][1]))
        & (col("l_returnflag") == lit(values_of_interest[3][2]))
    )
)

df_explicit_filter = df_lineitem.filter(filter_expr)

num_rows = df_explicit_filter.count()
print(
    f"Explicit filtering has number {num_rows} rows and took {time.time() - start_time} s"
)
start_time = time.time()

# Instead try a python UDF

def is_of_interest_impl(
    partkey_arr: pa.Array,
    suppkey_arr: pa.Array,
    returnflag_arr: pa.Array,
) -> pa.Array:
    result = []
    for idx, partkey in enumerate(partkey_arr):
        partkey = partkey.as_py()
        suppkey = suppkey_arr[idx].as_py()
        returnflag = returnflag_arr[idx].as_py()
        value = (partkey, suppkey, returnflag)
        result.append(value in values_of_interest)

    return pa.array(result)


is_of_interest = udf(
    is_of_interest_impl,
    [pa.int64(), pa.int64(), pa.utf8()],
    pa.bool_(),
    "stable",
)

df_udf_filter = df_lineitem.filter(
    is_of_interest(col("l_partkey"), col("l_suppkey"), col("l_returnflag"))
)

num_rows = df_udf_filter.count()
print(f"UDF filtering has number {num_rows} rows and took {time.time() - start_time} s")
start_time = time.time()

# Now use a user defined function but lean on the built in pyarrow array
# functions so we never convert rows to python objects.

# To see other pyarrow compute functions see
# https://arrow.apache.org/docs/python/api/compute.html
#
# It is important that the number of rows in the returned array
# matches the original array, so we cannot use functions like
# filtered_partkey_arr.filter(filtered_suppkey_arr).


def udf_using_pyarrow_compute_impl(
    partkey_arr: pa.Array,
    suppkey_arr: pa.Array,
    returnflag_arr: pa.Array,
) -> pa.Array:
    results = None
    for partkey, suppkey, returnflag in values_of_interest:
        filtered_partkey_arr = pc.equal(partkey_arr, partkey)
        filtered_suppkey_arr = pc.equal(suppkey_arr, suppkey)
        filtered_returnflag_arr = pc.equal(returnflag_arr, returnflag)

        resultant_arr = pc.and_(filtered_partkey_arr, filtered_suppkey_arr)
        resultant_arr = pc.and_(resultant_arr, filtered_returnflag_arr)

        if results is None:
            results = resultant_arr
        else:
            results = pc.or_(results, resultant_arr)

    return results


udf_using_pyarrow_compute = udf(
    udf_using_pyarrow_compute_impl,
    [pa.int64(), pa.int64(), pa.utf8()],
    pa.bool_(),
    "stable",
)

df_udf_pyarrow_compute = df_lineitem.filter(
    udf_using_pyarrow_compute(col("l_partkey"), col("l_suppkey"), col("l_returnflag"))
)

num_rows = df_udf_pyarrow_compute.count()
print(
    f"UDF filtering using pyarrow compute has number {num_rows} rows and took {time.time() - start_time} s"
)
start_time = time.time()

# In the next two examples we will use rust defined functions.
# This first one is the simplest way to write a rust function where
# there is no additional data we need to provide. In this example, we
# hard code the values of interest. But there can be many examples of
# UDFs where you do not need any additional data beyond what is in
# the expressions.

udf_using_custom_rust_fn = udf(
    tuple_filter_fn,
    [pa.int64(), pa.int64(), pa.utf8()],
    pa.bool_(),
    "stable",
)

df_udf_custom_rust_fn = df_lineitem.filter(
    udf_using_custom_rust_fn(col("l_partkey"), col("l_suppkey"), col("l_returnflag"))
)

num_rows = df_udf_custom_rust_fn.count()
print(
    f"UDF filtering using a custom rust function has number {num_rows} rows and took {time.time() - start_time} s"
)
start_time = time.time()

# Suppose you do need to provide some additional data. In this case it
# is the values we are looking for. The below code creates an instance
# of a python wrapped rust struct. When we create it, we pass in the
# values of interest. On the rust side we have implemented __call__
# on this object, so we can treat it just like a python function as
# far as the UDF below is concerned. One important note here is that
# in the `udf` call below we must provide `name`.

tuple_filter_class = TupleFilterClass(values_of_interest)

udf_using_custom_rust_fn_with_data = udf(
    tuple_filter_class,
    [pa.int64(), pa.int64(), pa.utf8()],
    pa.bool_(),
    "stable",
    name="tuple_filter_with_data"   
)

df_udf_custom_rust_fn_with_data = df_lineitem.filter(
    udf_using_custom_rust_fn_with_data(col("l_partkey"), col("l_suppkey"), col("l_returnflag"))
)

num_rows = df_udf_custom_rust_fn_with_data.count()
print(
    f"UDF filtering using a custom rust struct has number {num_rows} rows and took {time.time() - start_time} s"
)
start_time = time.time()


# Custom iterator


custom_iterator_class = TupleFilterDirectIterationClass(values_of_interest)

udf_custom_iterator = udf(
    custom_iterator_class,
    [pa.int64(), pa.int64(), pa.utf8()],
    pa.bool_(),
    "stable",
    name="tuple_filter_with_data"
)

df_custom_iterator = df_lineitem.filter(
    udf_custom_iterator(col("l_partkey"), col("l_suppkey"), col("l_returnflag"))
)

num_rows = df_custom_iterator.count()
print(
    f"UDF filtering using a custom iterator {num_rows} rows and took {time.time() - start_time} s"
)
start_time = time.time()
