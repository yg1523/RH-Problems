using LinearAlgebra
using ClassicalOrthogonalPolynomials, SingularIntegrals, QuadGK
using ContinuumArrays
P = Legendre()

n = 5
#normalize to use Cauchy Operator
stieltjes(P, 0.1+0im)[1:n]/(-2π*im)

x = ClassicalOrthogonalPolynomials.grid(P, n)
C⁺ = stieltjes(P, x .+ 0im)[:,1:n]/(-2π*im) 
C⁻ = stieltjes(P, x .- 0im)[:,1:n]/(-2π*im)

using Test
# evaluate the stiletjes transform of exp(x)
c = transform(P, exp) # Legendre coefficients
f = P*c # Legendre expansion
@test f[0.1] ≈ exp(0.1)

z = 0.1+0.2im
@test stieltjes(f, z) ≈ quadgk(x -> -exp(x)/(x-z), -1, 1)[1]

g = x -> -exp(x)/(x-z)
@time quadgk(g, -1, 1)[1]
@time stieltjes(f, z)







#Application of ContinuumArrays
basis = Legendre()
axes(basis, 1)

x = range(-1, 1, 100)
Q = basis[x, 1:5] #construct a quasimatrix
f = x -> x^2
c = Q\f.(x) #coefficient of approximation

f_pred = Q * c  # 逼近值
f_true = f.(x)

using Plots
plot(x, f_true, label="True f(x)")
plot!(x, f_pred, label="Approximation")


F = ClassicalOrthogonalPolynomials.Fourier()
f2(x) = sin(x) + 0.5cos(2x)
coeffs = expand(F, f2)

N = 200
truncated_coeffs = coeffs[1:N]
f_approx = F[:, 1:N] * truncated_coeffs
f_approx[1.5]
f2(1.5)

P = Legendre()                     # Legendre 多项式
x = ClassicalOrthogonalPolynomials.grid(P, 5)                      # 选择5个格点
B = P[:, 1:5]                       # 构造 quasimatrix：5个 Legendre 基底在这5点的值

D = ContinuumArrays.Derivative(-1..1)              # 一阶导数算子
DB = D * B 








#Application of ApproxFun
using ApproxFun
F = Fun(x -> x^2, ApproxFun.Chebyshev())
ApproxFun.coefficients(F)
ApproxFun.transform(ApproxFun.Chebyshev(), values(F)) ≈ ApproxFun.coefficients(F)



using ApproxFun
using SpecialFunctions
a, b = -20, 10
d = a..b

x = Fun(d)
D = ApproxFun.Derivative(d)
L = D^2 - x

B = Dirichlet(d)
B_vals = [airyai(a), airyai(b)]
u = [B; L] \ [B_vals, 0]

import Plots
Plots.plot(u, xlabel="x", ylabel="u(x)", legend=false)




using ApproxFun
using LinearAlgebra
a, b = 0, 2pi
d = a..b
D = ApproxFun.Derivative(d)
L = D^2 + 4

A = [L; ivp()]
u0 = 0
dtu0 = 2

u = \(A, [0,u0,dtu0], tolerance=1e-6)
t = a:0.1:b
Plots.plot(t, u.(t), xlabel="t", label="u(t)")
Plots.plot!(t, sin.(2t), label="Analytical", seriestype=:scatter)