# Datafusion Rust UDF Example

This repository is a minimum demonstration of a user being able to write their
own UDF functions in rust and incorporate them into a datafusion project.

# How to run

To run the example shown in `python-example` you will need to have a working
version of `datafusion-python` repository. This quick example doesn't copy the
requirements from that repository, but if you have it's working environment set
up then you should be able to build this code with

```shell
maturin develop --release
```

You will need a copy of the `lineitem.parquet` file that is generated within
datafusion-python repository under the examples. Place this file in
`./python-example`.
