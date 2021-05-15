import jax.numpy as jnp
from jax import jit, grad, random, lax, make_jaxpr, random
import math
import time

from context import Context

def dense(params, nin, nout, bias=True, dtype=jnp.float32):
    w = params('w', [nin, nout], dtype)
    b = params('b', [nout], dtype)
    def forward(x):
        y = jnp.dot(x, w)
        if bias:
            y += b
        return y
    return forward


def dense(params, nin, nout, bias=True, dtype=jnp.float32, name='dense'):
    w = params(name, 'w',  shape=[nin, nout], dtype=dtype)
    if bias: b = params(name, 'b', shape=[nout], dtype=dtype)
    return lambda x: jnp.dot(x, w) + b if bias else jnp.dot(x, w)

def conv2d(ctx, x, nin, nout, filter_shape, strides=None, padding='VALID', bias=True, dtype=jnp.float32):
    kernel_shape = (*filter_shape, nin, nout)
    one = (1,) * len(filter_shape)
    strides = strides or one

    w = ctx.var('w', kernel_shape, dtype)
    y = lax.conv_general_dilated(x, w, strides, padding, one, one, dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if bias:
        b = ctx.var('b', (nout, ), dtype)
        y += b
    return y


def f(trainables, nontrainables, x, reuse, nin):
    ctx = Context(trainables, nontrainables, reuse)
    prod = ctx.var('prod', [], jnp.float32, False)

    dense1 = dense(params('dense_1'), nin, 10)
    dense2 = dense(tx('dense_2'), nin, 10)

    return lax.cond(prod < 0, lambda _: dense1(x), lambda _: dense2(x), None)

key = random.PRNGKey(0)

trainables = dict()
nontrainables = dict()

n, nin = 256, 8192

x = jnp.ones((n, nin))

s = time.time()
y = f(trainables, nontrainables, x, False, nin)
y.block_until_ready()
e = time.time()
print(e - s)

g = jit(f, static_argnums=(3, 4))

for _ in range(10):
    trainables = {name: random.normal(key, value.shape) for name, value in trainables.items()}
    s = time.time()
    y = g(trainables, nontrainables, x, True, nin)
    y.block_until_ready()
    e = time.time()
    print(e - s)
