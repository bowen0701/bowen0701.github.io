---
author: Bowen Lee
date: '2026-08-16'
layout: post
tags:
- ai-rl-infra
- machine-learning
- numpy
- pytorch
- dispatch
title: NumPy-PyTorch Dispatch Pattern
---
<details class="toc-details" markdown="1">
<summary><b>Table of Contents</b></summary>

* TOC
{:toc}

</details>


# NumPy-PyTorch Dispatch Pattern

For research-oriented GitHub repos: a simple pattern for framework-agnostic ML code.

## Pattern

```text
foo(x, ...)                         # Public API: only interface users call
    |
    isinstance(x, ...)              # Dispatch
    |
    |-- _foo_numpy(x, ...)          # Reference: executable math spec
    `-- _foo_pytorch(x, ...)        # Differentiable / accelerated version
```

```python
def _foo_numpy(x, ...):
    """Reference implementation used for correctness tests."""
    ...

def _foo_pytorch(x, ...):
    """PyTorch implementation used for training/autograd."""
    ...

def foo(x, ...):
    """Public API: dispatches based on input type."""
    if isinstance(x, np.ndarray):
        return _foo_numpy(x, ...)
    elif isinstance(x, torch.Tensor):
        return _foo_pytorch(x, ...)
    raise TypeError(f"Unsupported type: {type(x)}")
```

**Core principle:** One algorithm = one module. NumPy is the executable math spec; PyTorch is the differentiable implementation. Keeping them side-by-side makes comparison and debugging easy.

**Naming:** Underscore prefix (`_foo_numpy`, `_foo_pytorch`) signals implementation detail. `foo()` is the only public interface.

**Multi-arg dispatch:** Validate that all operands use the same backend. Fail fast on mixed `ndarray` + `Tensor` inputs.

## Testing

NumPy is the correctness reference:

```python
expected = _foo_numpy(x_numpy)
actual = _foo_pytorch(torch.from_numpy(x_numpy))
np.testing.assert_allclose(actual.detach().numpy(), expected)
```

Loop:

```text
paper -> NumPy -> correctness test -> PyTorch -> NumPy == PyTorch test -> training
```

## When to Split Files

**Keep together** by default:

- <= 2 backends
- short implementations
- educational / research code
- visual comparison matters

**Split** when:

- module grows beyond ~300-500 LOC
- third backend appears, e.g. JAX
- backend complexity exceeds algorithm complexity

```text
module/
|-- api.py              # public dispatch
|-- numpy_impl.py
|-- pytorch_impl.py
`-- jax_impl.py
```

## Alternatives

| Approach | Mechanism | Pros | Cons | Best for |
|---|---|---|---|---|
| **manual dispatch via `isinstance`** | Manual type check | Explicit, easy to debug, implementations stay visible | Boilerplate per function | Research / learning repos |
| **`functools.singledispatch`** | Decorator-based type dispatch | Stdlib, Pythonic, extensible | Implementations become scattered | Small Python libraries |
| **`array-api-compat`** | Unified namespace via Python Array API standard | One implementation across array libraries | Hides backend details; external dependency | Library interoperability |

No major ML framework uses `functools.singledispatch` for core dispatch. It is a stdlib convenience, not a framework architecture pattern.

`array-api-compat` is different: real scientific Python libraries use it for array interoperability. SciPy uses it in its experimental Array API support, and scikit-learn vendors it for estimators/metrics that can run against Array API compatible inputs. This is still not "framework core dispatch" like PyTorch's dispatcher; it is a practical way for array-consuming libraries to share one implementation across NumPy-like backends.

---

## Industry Dispatch Patterns

Industry systems also dispatch, but they solve a different problem.

This note's pattern answers:

> "I have one algorithm and want NumPy + PyTorch implementations that are easy to compare."

Framework dispatch answers:

> "I have thousands of ops, devices, dtypes, transforms, compiler passes, and custom kernels."

Do not copy framework architecture into a small research repo. Copy the principle:

> make the dispatch axis explicit.

### Dispatch Axes

| Layer               | Dispatch axis                                          | Mechanism                 | Use when                                                      |
| ------------------- | ------------------------------------------------------ | ------------------------- | ------------------------------------------------------------- |
| **Manual dispatch** | Input backend: `np.ndarray` vs `torch.Tensor`          | `isinstance`              | Small research code, educational repos, paper implementations |
| **Array API**       | Array namespace: NumPy, PyTorch, JAX, CuPy             | `array_namespace(x)`      | You want one implementation across array libraries            |
| **PyTorch**         | Tensor device, dtype, layout, autograd, compiler modes | Internal dispatcher       | You are implementing a tensor framework or runtime            |
| **JAX**             | Transformation context: `jit`, `grad`, `vmap`          | Tracers + primitive rules | You want transformations over pure functions                  |
| **Triton**          | Kernel signature: shapes, strides, constants, hardware | JIT compile + cache       | You are writing GPU kernels                                   |

In JAX, "transformation context" means the wrapper currently interpreting the function. JAX traces primitive array operations such as add, multiply, matmul, reshape, and sum, then gives those operations different meanings depending on the active wrapper:

```python
f(x)             # normal eager execution
jax.jit(f)(x)    # trace and compile f before running it
jax.grad(f)(x)   # trace f to build a derivative function
jax.vmap(f)(xs)  # trace f once and apply it across a batch dimension
```

### What to Copy

Copy the idea that every dispatch system has one primary question:

| System          | Primary question                                      |
| --------------- | ----------------------------------------------------- |
| Manual dispatch | "Is this a NumPy array or PyTorch tensor?"            |
| Array API       | "Which array namespace should this function use?"     |
| PyTorch         | "Which kernel or mode should this tensor op use?"     |
| JAX             | "Which transformation is interpreting this function?" |
| Triton          | "Which compiled GPU kernel matches this signature?"   |

That is the transferable lesson. The machinery is not.

### What Not to Copy

Do not copy PyTorch's dispatcher, JAX's tracing model, or Triton's JIT cache into a small repo. Those systems exist because framework authors need to handle:

- thousands of ops
- many devices and dtypes
- autograd and compiler modes
- custom kernels
- graph transformations
- binary compatibility

For small research libraries, this usually means avoiding:

- registry / protocol / adapter / factory hierarchies
- custom `Tensor` wrappers
- reimplementing `dtype`, `shape`, broadcasting, or autograd

### Practical Rule

Use the simplest dispatch layer that matches the actual variability:

| Variability                                       | Pattern                           |
| ------------------------------------------------- | --------------------------------- |
| NumPy vs PyTorch implementation                   | Manual dispatch with `isinstance` |
| Same math maps cleanly across array libraries     | Array API namespace               |
| Many ops, devices, dtypes, autograd modes         | Framework dispatcher              |
| Function transformations are the main abstraction | JAX tracing                       |
| Hardware-specialized kernels                      | Triton JIT                        |

For this note's target use case, the right answer is usually:

> manual public dispatch + private NumPy/PyTorch implementations + NumPy-reference tests.

Use `array-api-compat` only when one shared implementation is more valuable than seeing separate NumPy and PyTorch code.

## References

- Data APIs Consortium. [Python Array API standard](https://data-apis.org/array-api/latest/).
- array-api-compat. [array-api-compat documentation](https://data-apis.org/array-api-compat/).
- Yang (2020). [Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/).
- PyTorch. [PyTorch dispatcher walkthrough](https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough).
- JAX. [Autodidax: JAX core from scratch](https://docs.jax.dev/en/latest/autodidax.html).
- JAX. [JAX primitives](https://docs.jax.dev/en/latest/jax-primitives.html).
