#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // std::vector <-> Python list
namespace py = pybind11;

#include <iostream>
#include <vector>
#include <stdexcept>

#include "TensorX/Tensor.hpp"   // путь поправь под свой include

int add_ints(int a, int b){
    return a + b;
}

static tensor::Tensor<float> from_1d(const py::sequence& seq) {
    int n = (int) seq.size();
    std::vector<int> dims = {n};
    tensor::Tensor<float> t(1, dims.data());
    for (int i = 0; i < n; ++i) {
        t.get(i) = py::cast<float>(seq[i]);
    }
    return t;
}

static tensor::Tensor<float> from_2d(const py::sequence& seq) {
    int rows = (int) seq.size();
    if (rows == 0) throw std::runtime_error("from_list: empty 2D list");

    py::object first_row_obj = seq[0];
    if (!py::isinstance<py::sequence>(first_row_obj))
        throw std::runtime_error("from_list: expected list of lists");

    py::sequence first_row = first_row_obj.cast<py::sequence>();
    int cols = (int)first_row.size();
    if (cols == 0) throw std::runtime_error("from_list: empty row in 2D list");

    // check rectangular
    for (int r = 0; r < rows; ++r) {
        py::object row_obj = seq[r];
        if (!py::isinstance<py::sequence>(row_obj))
            throw std::runtime_error("from_list: expected list of lists");
        py::sequence row = row_obj.cast<py::sequence>();
        if ((int)row.size() != cols)
            throw std::runtime_error("from_list: all rows must have same length");
    }

    int dims_arr[2] = {rows, cols};
    tensor::Tensor<float> t(2, dims_arr);

    for (int r = 0; r < rows; ++r) {
        py::sequence row = seq[r].cast<py::sequence>();
        for (int c = 0; c < cols; ++c) {
            t.get(r, c) = py::cast<float>(row[c]);
        }
    }
    return t;
}

static std::vector<int> parse_indices(const tensor::Tensor<float>& t, py::args args) {
    int rank = t.getRank();
    if ((int)args.size() != rank) {
        throw std::runtime_error(
            "Expected exactly " + std::to_string(rank) + " indices, got " + std::to_string(args.size())
        );
    }

    std::vector<int> idxs;
    idxs.reserve(args.size());

    int* dims = t.getDims(); // null if rank==0
    for (int i = 0; i < (int)args.size(); ++i) {
        if (!py::isinstance<py::int_>(args[i])) {
            throw std::runtime_error("All indices must be int");
        }
        int idx = args[i].cast<int>();
        if (idx < 0) {
            throw std::runtime_error("Index must be >= 0");
        }
        if (rank > 0 && idx >= dims[i]) {
            throw std::runtime_error(
                "Index out of bounds at axis " + std::to_string(i) +
                ": idx=" + std::to_string(idx) +
                ", dim=" + std::to_string(dims[i])
            );
        }
        idxs.push_back(idx);
    }
    return idxs;
}

static int linear_index_row_major(const tensor::Tensor<float>& t, const std::vector<int>& idxs) {
    int rank = t.getRank();
    if (rank == 0) return 0;

    int* dims = t.getDims();
    int index = 0;
    int stride = 1;

    for (int d = rank - 1; d >= 0; --d) {
        if (d != rank - 1) {
            stride *= dims[d + 1];
        }
        index += idxs[d] * stride;
    }
    return index;
}

PYBIND11_MODULE(_tensorx, m) {
    m.doc() = "TensorX Python bindings";
    m.def("add_ints", &add_ints, "A test function");

    m.def("make_tensor", []() {
        tensor::Tensor<float> t(2, 3);   // использует шаблон + ctor
        std::cout << t << std::endl;
        return t.getRank();
    });

    // Scalar Tensor
    m.def("scalar", [](float v) {
        return tensor::scalar<float>(v);
    }, py::arg("value"));

    // From List
    m.def("from_list", [](py::object obj) {
            if (!py::isinstance<py::sequence>(obj))
                throw std::runtime_error("from_list: expected a sequence");

            py::sequence seq = obj.cast<py::sequence>();
            if (seq.size() == 0)
                throw std::runtime_error("from_list: empty list");

            // detect 2D if first element is sequence
            py::object first = seq[0];
            if (py::isinstance<py::sequence>(first)) {
                return from_2d(seq);
            } else {
                return from_1d(seq);
            }
        }, py::arg("data"));

    // --- Tensor<float> ---
    py::class_<tensor::Tensor<float>>(m, "Tensor")
        // 1) Tensor() -> scalar
        .def(py::init<>())

        // 2) Tensor(int rank, int dims[])
        .def(py::init([](int rank, std::vector<int> dims) {
            if (rank < 0) throw std::runtime_error("Tensor(rank, dims): rank must be >= 0");

            if (rank == 0) {
            // игнорируем dims, как в твоём Tensor(rank, dims[])
            return tensor::Tensor<float>(0, nullptr);
            }

            if ((int) dims.size() != rank) {
            throw std::runtime_error("Tensor(rank, dims): len(dims) must equal rank");
            }

            for (int d : dims) {
                if (d <= 0) {
                    throw std::runtime_error("Tensor(rank, dims): dims must be > 0");
                }
            }

            return tensor::Tensor<float>(rank, dims.data());
        }), py::arg("rank"), py::arg("dims"))

        // 3) Tensor(d1, d2, ...)
        .def(py::init([](py::args args) {
            if (args.size() == 0) {
                throw std::runtime_error("Tensor(*dims): provide at least one int");
            }
            std::vector<int> dims;
            dims.reserve(args.size());
            for (auto a : args) {
                if (!py::isinstance<py::int_>(a)) {
                    throw std::runtime_error("Tensor(*dims): all dims must be int");
                }
                int d = a.cast<int>();
                if (d <= 0) throw std::runtime_error("Tensor(*dims): dims must be > 0");
                dims.push_back(d);
            }
            return tensor::Tensor<float>((int)dims.size(), dims.data());
        }))

        // rank / length / dims
        .def_property_readonly("rank", &tensor::Tensor<float>::getRank)
        .def_property_readonly("length", &tensor::Tensor<float>::getLength)
        .def_property_readonly("dims", [](const tensor::Tensor<float>& t) {
            std::vector<int> out;
            out.reserve(t.getRank());
            int* d = t.getDims();
            for (int i = 0; i < t.getRank(); ++i) out.push_back(d[i]);
            return out;
        })

        .def("get", [](tensor::Tensor<float>& t, py::args idx_args) {
            // scalar: rank==0 => get() без индексов
            if (t.getRank() == 0) {
                if (idx_args.size() != 0) throw std::runtime_error("Scalar tensor: use get() with no indices");
                return t.getCoeffs()[0];
            }

            auto idxs = parse_indices(t, idx_args);
            int lin = linear_index_row_major(t, idxs);
            return t.getCoeffs()[lin];
        })

        .def("set", [](tensor::Tensor<float>& t, py::args args) {
            // scalar: set(value)
            if (t.getRank() == 0) {
                if (args.size() != 1) throw std::runtime_error("Scalar tensor: use set(value)");
                float v = args[0].cast<float>();
                t.getCoeffs()[0] = v;
                return;
            }

            // rank>0: set(i,j,k,..., value) => last is value, rest are indices
            int rank = t.getRank();
            if ((int)args.size() != rank + 1) {
                throw std::runtime_error(
                    "Expected " + std::to_string(rank) + " indices + value, got " + std::to_string(args.size())
                );
            }

            // split
            py::tuple idx_tuple(rank);
            for (int i = 0; i < rank; ++i) idx_tuple[i] = args[i];

            py::args idx_args(idx_tuple); // create args-like view
            auto idxs = parse_indices(t, idx_args);
            int lin = linear_index_row_major(t, idxs);

            float v = args[rank].cast<float>();
            t.getCoeffs()[lin] = v;
        })

        // nice printing in Python: print(t)
        .def("__repr__", [](const tensor::Tensor<float>& t) {
            // use your toString()
            return t.toString();
        });

}
