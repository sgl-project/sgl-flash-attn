/*Adapt from:
https://github.com/neuralmagic/vllm-flash-attention/blob/90eacc1af2a7c3de62ea249e929ed5faccf38954/csrc/common/pytorch_shim.h
  Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <cmath>
#include <limits>
#include <optional>

/**
 * Unfortunately, the type signatures of the flash_attn ops are not compatible
 * with the PyTorch library bindings. To get around that we use
 * `pytorch_shim` which exposes the API using
 * PyTorch compatible types to the types, then converts them to the types
 * expected by the flash_attn ops. This shims allows us to make minimal changes
 * to `flash_api.cpp` making it easier to synchronize with upstream changes.
 *
 * The `pytorch_library_compatible_type` struct is used to map from the
 * flash_attn ops types to a PyTorch library compatible one. The main issues is
 * that the following types are not support by PyTorch library bindings:
 *  - `int`
 *  - `float`
 *  - `std::optional<T> &`
 *  - `std::optional<const torch::stable::Tensor> &`
 * So we convert them to (respectively):
 *  - `int64_t`
 *  - `double`
 *  - `const std::optional<T>&`
 *  - `const std::optional<torch::stable::Tensor>&`
 */

template <typename T>
struct pytorch_library_compatible_type {
  using type = T;
  static T convert_from_type(T arg) {
    return arg;
  }
};

template <typename T>
using pytorch_library_compatible_type_t = typename pytorch_library_compatible_type<T>::type;

template <typename T>
T convert_from_pytorch_compatible_type(pytorch_library_compatible_type_t<T> arg) {
  return pytorch_library_compatible_type<T>::convert_from_type(arg);
}

// Map `std::optional<T> &` -> `const std::optional<T>&`
//  (NOTE: this is bit unsafe but non of the ops in flash_attn mutate
//   the optional container)
template <typename T>
struct pytorch_library_compatible_type<std::optional<T>&> {
  using type = const std::optional<T>&;
  static std::optional<T>& convert_from_type(const std::optional<T>& arg) {
    return const_cast<std::optional<T>&>(arg);
  }
};

// Map `std::optional<T>` ->
//          `std::optional<pytorch_library_compatible_type_t<T>>`
//  (NOTE: tested for `std::optional<int>` -> `std::optional<int64_t>`)
template <typename T>
struct pytorch_library_compatible_type<std::optional<T>> {
  using type = std::optional<pytorch_library_compatible_type_t<T>>;
  static std::optional<pytorch_library_compatible_type_t<T>> convert_from_type(std::optional<T> arg) {
    return arg;
  }
};

// Map `std::optional<const torch::stable::Tensor>&` -> `const std::optional<torch::stable::Tensor>&`
template <>
struct pytorch_library_compatible_type<std::optional<const torch::stable::Tensor>&> {
  using type = const std::optional<torch::stable::Tensor>&;
  static std::optional<const torch::stable::Tensor>& convert_from_type(const std::optional<torch::stable::Tensor>& arg) {
    return const_cast<std::optional<const torch::stable::Tensor>&>(reinterpret_cast<const std::optional<const torch::stable::Tensor>&>(arg));
  }
};

// Map `int` -> `int64_t`
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    STD_TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "int64_t value is too large to be converted to int");
    STD_TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "int64_t value is too small to be converted to int");
    return arg;
  }
};

// Map `float` -> `double`
template <>
struct pytorch_library_compatible_type<float> {
  using type = double;
  static float convert_from_type(double arg) {
    STD_TORCH_CHECK(
        std::abs(arg) <= std::numeric_limits<float>::max(), "double value is too large to be converted to float");
    return arg;
  }
};

//
//  Shim Utils
//

template <auto fun>
struct pytorch_shim;

template <typename Ret, typename... Args, Ret (*fun)(Args...)>
struct pytorch_shim<fun> {
  static Ret call(pytorch_library_compatible_type_t<Args>... args) {
    return fun(convert_from_pytorch_compatible_type<Args>(args)...);
  }
};
