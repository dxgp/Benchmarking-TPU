import os
import torch
import numpy as np
import timeit
import functools
from prettytable import PrettyTable


START=100
END=200
GAP = 20


def helper_func(shapes,pytorch_func,display_name=""):
    ts = [torch.randn(x, device="cpu",requires_grad=True) for x in shapes]
    out = pytorch_func(*ts)
    out.mean().backward()
    torch_fp = timeit.Timer(functools.partial(pytorch_func, *ts)).timeit(5) * 1000/5
    torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), pytorch_func, ts)).timeit(5) * 1000/5
    print("testing(%s)  %40r    torch fp: %.2fms  bp: %.2fms" % (display_name,shapes, torch_fp, torch_fbp-torch_fp))
    return [display_name,shapes, str(torch_fp)[:6], str(torch_fbp-torch_fp)[:6]]

"""#*******Display results in a table********
def display_results(results):
    x = PrettyTable()
    x.field_names = ["Operation","Shapes","Forward Prop","Backprop"]
    for result in results:
        x.add_row(result)
    print(x)"""

#**********TEST FUNCTIONS**********
def stress_test(func_name,stress_level = 10):
    results = []
    for ordermag in range(stress_level):
        offset = np.random.randint(10**ordermag,10**(ordermag+1))
        results.append(func_name(offset))
    display_results(results)


def test_add(dim):
    return helper_func([(dim,dim), (dim,dim)], lambda x,y: x+y,"Add")
def test_sub(dim):
    return helper_func([(dim,dim), (dim,dim)], lambda x,y: x-y,"Sub")
def test_mul(dim):
    return helper_func([(dim,dim), (dim,dim)], lambda x,y: x*y,"Multiply")
def test_div(dim):
    return helper_func([(dim,dim), (dim,dim)], lambda x,y: x/y,"Divide")
def test_pow(dim):
    return helper_func([(dim,dim), (dim,dim)], lambda x,y: x**y,"Power")
def test_sqrt(dim):
    return helper_func([(dim,dim)], lambda x: x.sqrt(),"Sqrt")
def test_relu(dim):
    return helper_func([(dim,dim)], lambda x: x.relu(),"ReLU")
def test_leakyrelu(dim):
    return helper_func([(dim,dim)], lambda x: torch.nn.functional.leaky_relu(x,0.01),"LReLU")
def test_abs(dim):
    return helper_func([(dim,dim)], lambda x: torch.abs(x),"Abs")
def test_log(dim):
    return helper_func([(dim,dim)], lambda x: torch.log(x),"Log")
def test_exp(dim):
    return helper_func([(dim,dim)], lambda x: torch.exp(x),"Exp")
def test_sign(dim):
    return helper_func([(dim,dim)], lambda x: torch.sign(x),"Sign")
def test_sigmoid(dim):
    return helper_func([(dim,dim)], lambda x: x.sigmoid(),"Sigmoid")
def test_softplus(dim):
    return helper_func([(dim,dim)], lambda x: torch.nn.functional.softplus(x),"Softplus")
def test_relu6(dim):
    helper_func([(dim,dim)], lambda x: torch.nn.functional.relu6(x),"ReLU6")
def test_hardswish(dim):
    helper_func([(dim,dim)], lambda x: torch.nn.functional.hardswish(x),"Hardswish")
#mish(x) =  x*tanh(softplus(x))
def test_mish(dim):
    def _calc_mish(x):
        return x*torch.tanh(torch.nn.functional.softplus(x))
    return helper_func([(dim,dim)],_calc_mish,"mish")
def test_dot(dim):
    dim1 = np.random.randint(dim//2,dim)
    return helper_func([(dim,dim1),(dim1,dim)],lambda x,y: x.matmul(y),"matmul")

#multiple for loops, gonna have to break it up?
def test_dot2D(dim):
    dim1 = np.random.randint(dim//2,dim)
    return helper_func([(dim,dim1),(dim1,dim)],lambda x,y: x @ y,"dot2D")

def test_dot3D(dim):
    dim1 = np.random.randint(dim//2,dim)
    dim3 = np.random.randint(dim//2,dim)
    return helper_func([(dim3,dim,dim1),(dim3,dim1,dim)],lambda x,y: x @ y,"dot3D")

def test_dot4D(dim):
    results = []
    gdim = 100
    dim3 = np.random.randint(gdim//2,gdim)
    dim4 = np.random.randint(gdim//16,gdim//8)
    return helper_func([(dim4,dim3,dim,dim1),(dim4,dim3,dim1,dim)],lambda x,y: x @ y,"dot4D")


"""def test_sum(dim):
    # modify -- does not work properly
    results = []
    dim1 = np.random.randint(END//2,END)
    dim2 = np.random.randint(END//2,END)
    dim3 = np.random.randint(END//2,END)
    dim4 = np.random.randint(END//2,END)
    for dim in range(dim):
        results.append(helper_func([(dim2,dim1)],lambda x: x.sum()),"sum2D")
    for dim in range(dim):
        results.append(helper_func([(dim4,dim3,dim2,dim1)],lambda x: x.sum(axis=(1,2)),"sum4D,(1,2)"))
    for dim in range(dim):
        results.append(helper_func([(dim4,dim3,dim2,dim1)],lambda x: x.sum(axis=(1)),"sum4D,(1)"))
    display_results(result)"""
def test_mean_axis(dim):
    return helper_func([(dim,2*dim,3*dim,4*dim)],lambda x: x.mean(axis=(1,2)),"Mean")
def test_logsoftmax(dim):
    return helper_func([(dim,dim)],lambda x: torch.nn.LogSoftmax(dim=1)(x),"LogSoftmax")
def test_tanh(dim):
    return helper_func([(dim,dim)],lambda x: x.tanh()),"Tanh"
def test_topo_sort(dim):
    return helper_func([(dim,dim)],lambda x: (x+x)*x),"Topo Sort??"
def test_scalar_mul(dim):
    scalar_val = np.random.randint()
    return helper_func([(dim,dim)],lambda x: x*scalar_val,"Scalar Mult")
def test_scalar_rmul(dim):
    scalar_val = np.random.randint()
    return helper_func([(dim,dim)],lambda x: scalar_val*x,"Reverse Scalar Mult")
def test_scalar_sub(dim):
    scalar_val = np.random.randint()
    return helper_func([(dim,dim)],lambda x: x-scalar_val,"Scalar Subtraction")
def test_scalar_rsub(dim):
    scalar_val = np.random.randint()
    return helper_func([(dim,dim)],lambda x: scalar_val-x,"Reverse Scalar Mult")
def test_slice(dim):
    random_slice = np.random.randint(START,END)
def test_pad2d(dim):
    return helper_func([(dim,dim,dim,dim)],lambda x: torch.nn.functional.pad(x,(1,2,3,4)),"Pad2D")
def test_transpose(dim):
    return helper_func([(dim,dim,dim,dim)],lambda x: x.movedim((3,2,1,0),(0,1,2,3)),"Transpose")
def test_reshape(dim):
    return helper_func([(dim//4,dim//2,dim,dim)],lambda x: torch.reshape(x,(-1,dim//2,dim,dim)),"Reshape")

# gonna have to come back to these, need to take another look at the shapes
def test_conv2d(bs,cin,groups,H,W):
    return helper_func([bs,cin,11,])
def test_strided_conv2d():
    pass
def test_maxpool_2d():
    pass
def test_avgpool2d():
    pass

if __name__=='__main__':
    stress_test(test_reshape)